#!/bin/bash
set -v
set -e
set -x
exp_no=dcps_fewshot
GPU=2
dataset=(   Aircraft  Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars  SUN397)
lr=(2e-3 2e-3 2e-3 3e-3 3e-3 2e-3 2e-3 3e-3 2e-3 1e-3 2e-3)
batch_size=(16 16 16 16 16 16 16 16 16 16 16)
hard_sample_ratio=(0.4 0.15 0.3 0.3 0.1 0.1 0.15 0.1 0.1 0.2 0.4)
threshold_percentile=(0.1 0.01 0.03 0.1 0.02 0.03 0.05 0.02 0.1 0.07 0.1)
prompt_length=2
prompt_depth=12



CUDA_VISIBLE_DEVICES=${GPU} python3 -m src.main \
    --train-mode=prompt \
    --train-dataset=${dataset[0]} \
    --eval-dataset=${dataset[0]} \
    --lr=${lr[0]} \
    --iterations 500 \
    --few-shot 5 \
    --eval-interval 500 \
    --hard_sample_ratio=${hard_sample_ratio[0]} \
    --threshold_percentile=${threshold_percentile[0]} \
    --save ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth} \
    --batch-size ${batch_size[0]} \
    --batch-size-eval ${batch_size[0]}\
    --prompt_width ${prompt_length}\
    --prompt_depth_vision ${prompt_depth}\
    --prompt_depth_text   ${prompt_depth}\
    --trainer=DCPS
#
#for ((i = 1; i < ${#dataset[@]}; i++)); do
#    dataset_cur=${dataset[i]}
#    dataset_pre=${dataset[i - 1]}
#
#    # continue training
#    CUDA_VISIBLE_DEVICES=${GPU} python3 -m src.main \
#        --train-mode=prompt \
#        --train-dataset=${dataset_cur} \
#        --eval-dataset=${dataset_cur} \
#        --lr=${lr[i]} \
#        --iterations 500 \
#        --eval-interval 500 \
#        --few-shot 5 \
#        --hard_sample_ratio=${hard_sample_ratio[i]} \
#        --threshold_percentile=${threshold_percentile[i]} \
#        --save ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth} \
#        --load ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth}/${dataset_pre}.pth \
#        --batch-size ${batch_size[i]} \
#        --batch-size-eval ${batch_size[i]}\
#        --prompt_width ${prompt_length}\
#        --prompt_depth_vision ${prompt_depth}\
#        --prompt_depth_text   ${prompt_depth}\
#        --trainer=DCPS
#done
#
#
#python3 -m src.general_eval --eval_names exp_${exp_no}_${prompt_length}_${prompt_depth}\
#        --prompt_width ${prompt_length} \
#        --prompt_depth_vision ${prompt_depth}\
#        --prompt_depth_text   ${prompt_depth}
