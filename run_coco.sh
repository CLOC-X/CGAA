# train on coco
CUDA_VISIBLE_DEVICES=3,4,5,6 python scripts/dist_clip_coco.py --config /configs/coco_attn_reg.yaml
# inference on coco
python test_msc_flip_coco.py --model_path 