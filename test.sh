python test.py \
    --backbone swin_tiny \
    --dataset_file hico \
    --resume weights/FGAHOI_Tiny.pth \
    --pretrained param/swin_tiny*.pth \
    --num_verb_classes 117 \
    --num_obj_classes 80 \
    --output_dir logs \
    --merge \
    --hierarchical_merge \
    --task_merge \
    --eval \
    --hoi_path data/hico_20160224_det_6000ada

# python test.py \
#     --backbone swin_large_384 \
#     --dataset_file hico \
#     --resume weights/FGAHOI_Large.pth \
#     --pretrained param/swin_large*.pth \
#     --num_verb_classes 117 \
#     --num_obj_classes 80 \
#     --output_dir logs \
#     --merge \
#     --hierarchical_merge \
#     --task_merge \
#     --eval \
#     --hoi_path data/hico_20160224_det_6000ada \
#     --pretrain_model_path ""


    # --hoi_path data/hico_20160224_det \

# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
# --backbone swin_large_384 --dataset_file hico --resume weights/FGAHOI_Large.pth \
# --num_verb_classes 117 --num_obj_classes 80 --output_dir logs --epochs 150 --lr_drop 120 \
# --num_feature_levels 3 --num_queries 300 --merge --hierarchical_merge --task_merge --eval \
# --hoi_path data/gligen_hico_1 --pretrain_model_path "" --pretrained param/swin_large*.pth