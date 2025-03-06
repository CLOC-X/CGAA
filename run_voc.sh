# train on voc
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/dist_clip_voc.py --config /configs/voc_attn_reg.yaml
# inference on voc
python test_msc_flip_seg.py --model_path 

