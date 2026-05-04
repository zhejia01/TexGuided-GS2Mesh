#!/bin/bash

root_dir="/media/data8T/puhua/Synthetic4Relight/"
list="air_baloons chair hotdog jugs"
# list="jugs"
for i in $list
do
    # CUDA_VISIBLE_DEVICES=7 python train.py --eval \
    #     -s ${root_dir}${i} \
    #     -m /media/data8T/puhua/SyntheticlightOurs206full/${i}/3dgs \
    #     -c /media/data8T/puhua/remesh-2dgs-new0206-vox0.006new/${i}/save_obj/gsbingdinginit.ply \
    #     --iterations 15000 \
    #     --lambda_normal_render_depth 0.01 \
    #     --lambda_normal_smooth 0.02 \
    #     --lambda_mask_entropy 0.1 \
    #     --save_training_vis \
    #     --densify_grad_normal_threshold 1e-8 \
    #     --lambda_depth_var 1e-2

    # CUDA_VISIBLE_DEVICES=7 python eval_nvs.py --eval \
    #     -m /media/data8T/puhua/SyntheticlightOurs206full/${i}/3dgs \
    #     -c /media/data8T/puhua/SyntheticlightOurs206full/${i}/3dgs/chkpnt15000.pth

    CUDA_VISIBLE_DEVICES=7 python train.py --eval \
        -s ${root_dir}${i} \
        -m /media/data8T/puhua/SyntheticlightOurs206full/${i}/neilf \
        -c /media/data8T/puhua/SyntheticlightOurs206new/${i}/3dgs/chkpnt15000.pth \
        --save_training_vis \
        --position_lr_init 0 \
        --position_lr_final 0 \
        --normal_lr 0 \
        --sh_lr 0 \
        --opacity_lr 0 \
        --scaling_lr 0 \
        --rotation_lr 0 \
        --iterations 35000 \
        --lambda_base_color_smooth 1 \
        --lambda_roughness_smooth 0.5 \
        --lambda_light_smooth 1 \
        --lambda_light 0.01 \
        -t neilf --sample_num 64 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01
    
    CUDA_VISIBLE_DEVICES=7 python eval_nvs.py --eval \
        -m /media/data8T/puhua/SyntheticlightOurs206full/${i}/neilf \
        -c /media/data8T/puhua/SyntheticlightOurs206full/${i}/neilf/chkpnt35000.pth \
        -t neilf --skip_test

    CUDA_VISIBLE_DEVICES=7 python eval_relighting_syn4.py  \
       -m /media/data8T/puhua/SyntheticlightOurs206full/${i}/neilf   \
       -c /media/data8T/puhua/SyntheticlightOurs206full/${i}/neilf/chkpnt35000.pth  --sample_num 384
done