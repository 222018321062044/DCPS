#!/bin/bash
set -v
set -e
set -x

exp_no=dcps
GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(2e-3)
batch_size=(32 32 16 16 16 16 16 16 16 16 8)
prompt_length=2
prompt_depth=(8 10)


for ((j = 0; j < ${#prompt_depth[@]}; j++)); do
  CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
      --train-mode=prompt \
      --train-dataset=${dataset[0]} \
      --eval-dataset=${dataset[0]} \
      --lr=${lr[0]} \
      --iterations 1000 \
      --save ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth[j]} \
      --batch-size ${batch_size[0]} \
      --batch-size-eval ${batch_size[0]}\
      --prompt_width ${prompt_length}\
      --prompt_depth_vision ${prompt_depth[j]}\
      --prompt_depth_text   ${prompt_depth[j]}\
      --trainer=DCPS

  for ((i = 1; i < ${#dataset[@]}; i++)); do
      dataset_cur=${dataset[i]}
      dataset_pre=${dataset[i - 1]}

      # continue training
      CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
          --train-mode=prompt \
          --train-dataset=${dataset_cur} \
          --eval-dataset=${dataset_cur} \
          --lr=${lr[0]} \
          --iterations 1000 \
          --save ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth[j]} \
          --load ckpt/exp_${exp_no}_${prompt_length}_${prompt_depth[j]}/${dataset_pre}.pth \
          --batch-size ${batch_size[i]} \
          --batch-size-eval ${batch_size[i]}\
          --prompt_width ${prompt_length}\
          --prompt_depth_vision ${prompt_depth[j]}\
          --prompt_depth_text   ${prompt_depth[j]}\
          --trainer=DCPS
  done


  python -m src.general_eval --eval_names exp_${exp_no}_${prompt_length}_${prompt_depth[j]}\
          --prompt_width ${prompt_length} \
          --prompt_depth_vision ${prompt_depth[j]}\
          --prompt_depth_text   ${prompt_depth[j]}
done
