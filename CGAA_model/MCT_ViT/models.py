import torch
import torch.nn as nn
from functools import partial
from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F

import math

__all__ = ['deit_small_MCTformerPlus']

class MCTformerPlus(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 28*28 = 784
        self.num_patches = num_patches

        self.patch_size_large = to_2tuple(self.patch_embed.patch_size[0] * 2)

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_embed_pat_large = nn.Parameter(torch.zeros(1, num_patches//4, self.embed_dim))
        self.alpha = nn.Parameter(torch.zeros(1)+0.5)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        trunc_normal_(self.pos_embed_pat_large, std=.02)

        self.decay_parameter=decay_parameter

    def interpolate_pos_encoding(self, x, w, h, patchsize):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        patch_pos_embed = self.pos_embed_pat
        if npatch == N and w == h:
            return self.pos_embed_pat

        dim = x.shape[-1]

        w0 = w // patchsize
        h0 = h // patchsize
        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x, x_crop=None,n=12):
        if x_crop == None:
            x_crop = x
        B, nc, w, h = x.shape
        B_l, nc_l, w_l, h_l = x_crop.shape
        x_large = self.patch_embed_large(x_crop)
        # sorted_x_large, indices = torch.sort(x_large, -2, descending=True)
        # sorted_x_large, indices = torch.sort(sorted_x_large, -3, descending=True)
        # print(sorted_x_large)
        # x = x.index_select(-2, indices)
        x = self.patch_embed(x)

        pos_embed_pat = self.interpolate_pos_encoding(x, w, h, self.patch_embed.patch_size[0])
        pos_embed_pat_large = self.interpolate_pos_encoding(x_large, w_l, h_l, self.patch_size_large[0])
        x = x + pos_embed_pat
        x_large = x_large + pos_embed_pat_large


        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)
        x_large = torch.cat((cls_tokens, x_large), dim=1)

        x = self.pos_drop(x)
        x_large = self.pos_drop(x_large)
        attn_weights = []
        class_embeddings = []

        for i, blk in enumerate(self.blocks[0:len(self.blocks)//2]):
            x, weights_i = blk(x)
            x_large, weights_large_i = blk(x_large)
            attn_weights.append(weights_i)
            # attn_weights.append(weights_large_i)
            class_embeddings.append(x[:, 0:self.num_classes])
            class_embeddings.append(x_large[:, 0:self.num_classes])
        x, x_large = self.fuse_token(x, x_large)
        class_embeddings.append(x[:, 0:self.num_classes])
        class_embeddings.append(x_large[:, 0:self.num_classes])
        for i, blk in enumerate(self.blocks[len(self.blocks)//2:]):
            x, weights_i = blk(x)
            x_large, weights_large_i = blk(x_large)
            attn_weights.append(weights_i)
            # attn_weights.append(weights_large_i)
            class_embeddings.append(x[:, 0:self.num_classes])
            class_embeddings.append(x_large[:, 0:self.num_classes])


        return x[:, 0:self.num_classes], x_large[:, 0:self.num_classes], x[:, self.num_classes:],x_large[:, self.num_classes:], attn_weights, class_embeddings

    def forward(self, x, x_crop=None, return_att=False, n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]

        x_cls, x_large_cls, x_patch, x_patch_large, attn_weights, all_x_cls = self.forward_features(x, x_crop)
        n, p, c = x_patch.shape
        _, p_l, _ = x_patch_large.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
            x_patch_large = torch.reshape(x_patch_large, [n, w0//2, h0//2, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
            x_patch_large = torch.reshape(x_patch_large, [n, int(p_l ** 0.5), int(p_l ** 0.5), c])
        _x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch_large_ = x_patch_large.permute([0, 3, 1, 2]).contiguous()
        x_patch_large = F.interpolate(x_patch_large_, size=(_x_patch.shape[2], _x_patch.shape[3]), mode='bilinear', align_corners=False)
        x_patch = _x_patch + self.alpha * x_patch_large
        x_patch = self.head(x_patch)

        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)    # n, int(p ** 0.5)*int(p ** 0.5), c


        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

        x_patch_large = self.head(x_patch_large_)
        x_patch_large_flattened = x_patch_large.view(x_patch_large.shape[0], x_patch_large.shape[1], -1).permute(0, 2, 1)    # n, int(p ** 0.5)*int(p ** 0.5), c
        sorted_patch_token, indices = torch.sort(x_patch_large_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_large_flattened.size(-2) - 1,
                                  steps=x_patch_large_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_large_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

        x_cls_logits = x_cls.mean(-1)
        x_large_cls_logits = x_large_cls.mean(-1)
        x_cls_logits = (x_large_cls_logits + x_cls_logits)/2
        output = []
        output.append(x_cls_logits)
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)
        output.append(x_patch_large_logits)

        if return_att:
            feature_map = x_patch.detach().clone()  # B * C * 14 * 14
            feature_map = F.relu(feature_map)
            n, c, h, w = feature_map.shape

            attn_weights_ = torch.stack(attn_weights)  # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights_, dim=2)  # 12 * B * N * N
            mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
            patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]
            if attention_type == 'fused':
                cams = mtatt * feature_map  # B * C * 14 * 14
                cams = torch.sqrt(cams)
            elif attention_type == 'patchcam':
                cams = feature_map
            elif attention_type == 'mct':
                cams = mtatt
            else:
                raise f'Error! {attention_type} is not defined!'

            x_logits = (x_cls_logits + x_patch_logits) / 2
            return attn_weights_, cams, patch_attn, _x_patch
        else:
            return output

@register_model
def deit_small_MCTformerPlus(pretrained=False, **kwargs):
    model = MCTformerPlus(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model