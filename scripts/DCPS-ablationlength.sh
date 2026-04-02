#!/bin/bash
set -v
set -e
set -x

exp_no=dcps
GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(2e-3)
batch_size=(16 16 8 8 8 8 8 8 8 8 4)
prompt_length=(10 15 20)
prompt_depth=12


for ((j = 0; j < ${#prompt_length[@]}; j++)); do
  CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
      --train-mode=prompt \
      --train-dataset=${dataset[0]} \
      --eval-dataset=${dataset[0]} \
      --lr=${lr[0]} \
      --iterations 1000 \
      --save ckpt/exp_${exp_no}_${prompt_length[j]}_${prompt_depth} \
      --batch-size ${batch_size[0]} \
      --batch-size-eval ${batch_size[0]}\
      --prompt_width ${prompt_length[j]}\
      --n_ctx_vision ${prompt_length[j]}\
      --n_ctx_text ${prompt_length[j]}\
      --prompt_depth_vision ${prompt_depth}\
      --prompt_depth_text   ${prompt_depth}\
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
          --save ckpt/exp_${exp_no}_${prompt_length[j]}_${prompt_depth} \
          --load ckpt/exp_${exp_no}_${prompt_length[j]}_${prompt_depth}/${dataset_pre}.pth \
          --batch-size ${batch_size[i]} \
          --batch-size-eval ${batch_size[i]}\
          --prompt_width ${prompt_length[j]}\
          --n_ctx_vision ${prompt_length[j]}\
          --n_ctx_text ${prompt_length[j]}\
          --prompt_depth_vision ${prompt_depth}\
          --prompt_depth_text   ${prompt_depth}\
          --trainer=DCPS
  done

  python -m src.general_eval --eval_names exp_${exp_no}_${prompt_length[j]}_${prompt_depth}\
          --prompt_width ${prompt_length[j]} \
          --prompt_depth_vision ${prompt_depth}\
          --prompt_depth_text   ${prompt_depth}\
          --n_ctx_vision ${prompt_length[j]}\
          --n_ctx_text ${prompt_length[j]}
done
