#!/bin/bash
# Contamination robustness sweep on GPU 2.
# Runs 5 representative MVTec classes + MMS dataset at 5 contamination rates.
# Usage: bash run_contamination.sh [gpu_id]  (default: 2)

GPU=${1:-2}
mvtec_datapath=/datasets/pbonazzi/tinyglass_mvtec
mms_datapath=/datasets/pbonazzi/tinyglass_mmdataset
augpath=/datasets/pbonazzi/tinyglass_mvtec/dtd/images

mvtec_classes=('carpet' 'bottle' 'cable' 'metal_nut' 'transistor')
mms_classes=('mms_rpi')

cd ..

for rate in 0.0 0.05 0.10 0.20 0.30; do

    echo "=== [MVTec] contamination_rate=${rate} ==="
    mvtec_flags=($(for c in "${mvtec_classes[@]}"; do echo '-d '"${c}"; done))
    python main.py \
        --results_path "results/contamination/mvtec/rate_${rate}" \
        --gpu ${GPU} \
        --seed 0 \
        --test ckpt \
      net \
        -b resnet18 \
        -le layer2 \
        -le layer3 \
        --pretrain_embed_dimension 384 \
        --target_embed_dimension 384 \
        --patchsize 3 \
        --meta_epochs 50 \
        --eval_epochs 5 \
        --dsc_layers 2 \
        --dsc_hidden 512 \
        --pre_proj 1 \
        --mining 1 \
        --noise 0.015 \
        --radius 0.75 \
        --p 0.5 \
        --step 10 \
        --limit 392 \
      dataset \
        --distribution 2 \
        --mean 0.5 \
        --std 0.1 \
        --fg 0 \
        --rand_aug 1 \
        --batch_size 8 \
        --resize 256 \
        --imagesize 256 \
        --contamination_rate ${rate} \
        "${mvtec_flags[@]}" mvtec $mvtec_datapath $augpath

    echo "=== [MMS] contamination_rate=${rate} ==="
    mms_flags=($(for c in "${mms_classes[@]}"; do echo '-d '"${c}"; done))
    python main.py \
        --results_path "results/contamination/mms/rate_${rate}" \
        --gpu ${GPU} \
        --seed 0 \
        --test ckpt \
      net \
        -b resnet18 \
        -le layer2 \
        -le layer3 \
        --pretrain_embed_dimension 384 \
        --target_embed_dimension 384 \
        --patchsize 3 \
        --meta_epochs 50 \
        --eval_epochs 5 \
        --dsc_layers 2 \
        --dsc_hidden 512 \
        --pre_proj 1 \
        --mining 1 \
        --noise 0.015 \
        --radius 0.75 \
        --p 0.5 \
        --step 10 \
        --limit 392 \
      dataset \
        --distribution 2 \
        --mean 0.5 \
        --std 0.1 \
        --fg 0 \
        --rand_aug 1 \
        --batch_size 8 \
        --resize 256 \
        --imagesize 256 \
        --contamination_rate ${rate} \
        "${mms_flags[@]}" mvtec $mms_datapath $augpath

    echo "Rate ${rate} done at $(date)"
done

echo "All contamination runs finished at $(date)"
