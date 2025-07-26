import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
import cv2
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from CVF_model.PAR import PAR
from .Decoder.conv_head import LargeFOV, ASPP
from .MCT_ViT.models import deit_small_MCTformerPlus as MCT
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import PIL.Image as Image

def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()
class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q
def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)
def crf_inference(img, probs, t=10, scale_factor=1, labels=21):


    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


class CVF(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        self.in_channels = in_channels
        self.mct = MCT(num_classes=20,drop_rate=0.0,drop_path_rate=0.1,input_size=448)
        self.mct_path='XXX/checkpoint.pth'
        checkpoint = torch.load(self.mct_path, map_location='cpu')
        self.mct.load_state_dict(checkpoint['model'])

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)
        self.decoder_1 = LargeFOV(in_planes=self.mct.embed_dim, out_planes=self.num_classes,)
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.postprocessor = DenseCRF(iter_max=10,pos_xy_std=1,pos_w=3,bi_xy_std=67,bi_rgb_std=3,bi_w=4,)
        self.encoder.eval()
        self.alpha = nn.Parameter(torch.zeros(1)+0.5)
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True

    def _refine_cams(self,ref_mod, images, cams, valid_key):
        images = images.unsqueeze(0)
        cams = cams.unsqueeze(0)
        refined_cams = ref_mod(images.float(), cams.float())
        refined_label = refined_cams.argmax(dim=1)
        refined_label = valid_key[refined_label]

        return refined_label.squeeze(0)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;


        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    def load_cls_label_list(self,name_list_dir):
    
        return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()


    def forward(self, img, img_names='2007_000032', mode='train',cam_only=False,n_iter=100):
        cam_list = []
        patch_list = []
        b, c, h, w = img.shape
        self.encoder.eval()      
        attn_weights, cams_mct, patch_attn, x_patch = self.mct(img,return_att=True)  # #  attn_weights 12 * B * H * N * N
        patch_attn = torch.sum(patch_attn, dim=0)

        img_temp = img.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.
        self.iter_num += 1
        cam_labels_list = []
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)


        fts = self.decoder_fts_fuse(all_img_tokens)   
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)
        
        seg_1 = self.decoder_1(x_patch)
        seg = self.alpha * seg + (1-self.alpha) * seg_1


        x_patch = self.mct.head(x_patch)
        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)    # n, int(p ** 0.5)*int(p ** 0.5), c
        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=0.996).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
      
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]  # ([1, 400, 400])
            
            if self.iter_num > 20000 or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False
            cam_refined_list,patch, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn, patch_attn[i].unsqueeze(0),  
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)
            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)   # torch.Size([320, 320])
            cam_list.append(cam_labels)
            patch_list.append(patch)
        all_cam_labels = torch.stack(cam_list, dim=0)  # torch.Size([4, 320, 320])
        patch_attn_ = torch.stack(patch_list, dim=0) 
        refine_cams = torch.matmul(patch_attn_.unsqueeze(1), cams_mct.view(cams_mct.shape[0],cams_mct.shape[1], -1, 1)).reshape(cams_mct.shape[0],cams_mct.shape[1], h//16, w //16)

        return seg, all_cam_labels, attn_pred, x_patch_logits, refine_cams

        
