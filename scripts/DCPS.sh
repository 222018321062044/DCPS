#!/bin/bash
set -v
set -e
set -x
exp_no=dcps
GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
#dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
lr=(2e-3 2e-3 2e-3 3e-3 3e-3 2e-3 2e-3 3e-3 2e-3 1e-3 2e-3)
#lr=(1e-3 2e-3 3e-3 2e-3 2e-3 2e-3 2e-3 2e-3 3e-3 3e-3 2e-3)
batch_size=(64 64 64 64 64 64 64 64 64 32 32)
#batch_size=(32 64 64 64 64 32 64 64 64 64 64)
hard_sample_ratio=(0.4 0.15 0.3 0.3 0.1 0.1 0.15 0.1 0.1 0.2 0.4)
#hard_sample_ratio=(0.2 0.15 0.1 0.1 0.1 0.4 0.4 0.15 0.3 0.2 0.4)
threshold_percentile=(0.1 0.01 0.03 0.1 0.02 0.03 0.05 0.02 0.1 0.07 0.1)
#threshold_percentile=(0.07 0.05 0.02 0.1 0.03 0.1 0.1 0.01 0.1 0.02 0.03)
prompt_length=2
prompt_depth=12

CUDA_VISIBLE_DEVICES=${GPU} python3 -m src.main \
    --train-mode=prompt \
    --train-dataset=${dataset[0]} \
    --eval-dataset=${dataset[0]} \
    --lr=${lr[0]} \
    --iterations 2000 \
    --hard_sample_ratio=${hard_sample_ratio[0]} \
    --threshold_percentile=${threshold_percentile[0]} \
    --save ckpt/exp_${exp_no} \
    --batch-size ${batch_size[0]} \
    --batch-size-eval ${batch_size[0]}\
    --prompt_width ${prompt_length}\
    --prompt_depth_vision ${prompt_depth}\
    --prompt_depth_text   ${prompt_depth}\
    --trainer=DCPS

for ((i = 1; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python3 -m src.main \
        --train-mode=prompt \
        --train-dataset=${dataset_cur} \
        --eval-dataset=${dataset_cur} \
        --lr=${lr[i]} \
        --iterations 2000 \
        --hard_sample_ratio=${hard_sample_ratio[i]} \
        --threshold_percentile=${threshold_percentile[i]} \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}.pth \
        --batch-size ${batch_size[i]} \
        --batch-size-eval ${batch_size[i]}\
        --prompt_width ${prompt_length}\
        --prompt_depth_vision ${prompt_depth}\
        --prompt_depth_text   ${prompt_depth}\
        --trainer=DCPS
done


CUDA_VISIBLE_DEVICES=${GPU} python3 -m src.general_eval --eval_names exp_${exp_no} \
        --prompt_width ${prompt_length} \
        --prompt_depth_vision ${prompt_depth}\
        --prompt_depth_text   ${prompt_depth}

