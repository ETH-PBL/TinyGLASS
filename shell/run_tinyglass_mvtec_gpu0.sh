#!/bin/bash

datapath=/datasets/pbonazzi/tinyglass_mvtec
augpath=/datasets/pbonazzi/tinyglass_mvtec/dtd/images
classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --results_path results/tinyglass_mvtec \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b resnet18 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 384 \
    --target_embed_dimension 384 \
    --patchsize 3 \
    --meta_epochs 150 \
    --eval_epochs 5 \
    --dsc_layers 2 \
    --dsc_hidden 512 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 2 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 256 \
    --imagesize 256 "${flags[@]}" mvtec $datapath $augpath

echo "Done GPU0 at $(date)"
