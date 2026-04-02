#!/usr/bin/env python3
"""Inspect the contents of a saved DCPS checkpoint."""

import argparse
from pathlib import Path

import torch


IMPORTANT_KEYS = [
    'prompt_pool',
    'prototype_feature',
    'text_prompt_pool',
    'visual_prompt_pool',
    'scale_I_pool',
    'scale_T_pool',
    'proj_down_weight',
    'proj_down_bias',
    'proj_up_weight',
    'proj_up_bias',
    'ItoT_down_weight',
    'ItoT_down_bias',
    'ItoT_up_weight',
    'ItoT_up_bias',
    'TtoI_down_weight',
    'TtoI_down_bias',
    'TtoI_up_weight',
    'TtoI_up_bias',
]


def check_checkpoint(checkpoint_path: Path):
    print('=' * 70)
    print(f'Checking checkpoint: {checkpoint_path}')
    print('=' * 70)

    if not checkpoint_path.exists():
        print('ERROR: file not found.')
        return 1

    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f'\nFile size: {file_size_mb:.2f} MB')
    print('\nLoading checkpoint...')

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f'ERROR loading checkpoint: {exc}')
        return 1

    print('Checkpoint loaded successfully.')
    print('\n' + '-' * 70)
    print('CHECKPOINT STRUCTURE')
    print('-' * 70)

    if not isinstance(checkpoint, dict):
        print(f'Checkpoint type: {type(checkpoint).__name__}')
        return 0

    print(f'\nTotal keys: {len(checkpoint)}')
    print('\nAll keys:')
    for index, key in enumerate(sorted(checkpoint.keys()), start=1):
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            print(f'  {index:3d}. {key:<40} | Tensor {list(value.shape)!s:<20} | dtype: {value.dtype}')
        elif isinstance(value, dict):
            print(f'  {index:3d}. {key:<40} | Dict with {len(value)} keys')
        elif isinstance(value, list):
            print(f'  {index:3d}. {key:<40} | List with {len(value)} items')
        else:
            print(f'  {index:3d}. {key:<40} | Type: {type(value).__name__}')

    print('\n' + '-' * 70)
    print('IMPORTANT BUFFERS CHECK')
    print('-' * 70)

    valid_count = 0
    for key in IMPORTANT_KEYS:
        if key not in checkpoint:
            print(f'  {key:<30} | NOT FOUND')
            continue

        value = checkpoint[key]
        if not isinstance(value, torch.Tensor):
            print(f'  {key:<30} | Not a tensor (type: {type(value).__name__})')
            continue

        tensor_sum = value.abs().sum().item()
        status = 'VALID' if tensor_sum > 0 else 'EMPTY/ZERO'
        print(f'  {key:<30} | Shape: {list(value.shape)!s:<20} | Sum: {tensor_sum:.4f} | {status}')
        if tensor_sum > 0:
            valid_count += 1

    print(f'\nValid buffers found: {valid_count}/{len(IMPORTANT_KEYS)}')

    prompt_learner_keys = [key for key in checkpoint.keys() if 'prompt_learner' in key]
    print('\n' + '-' * 70)
    print('PROMPT_LEARNER PARAMETERS')
    print('-' * 70)
    if prompt_learner_keys:
        for key in sorted(prompt_learner_keys):
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f'  {key:<60} | {list(value.shape)!s:<20}')
    else:
        print('  No prompt_learner keys found')

    print('\n' + '-' * 70)
    print('STATE_DICT CHECK')
    print('-' * 70)
    if 'state_dict' in checkpoint:
        print("  Contains 'state_dict' key")
        print(f"  state_dict has {len(checkpoint['state_dict'])} keys")
    else:
        print("  No 'state_dict' key (checkpoint may already be a state dict)")

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    if valid_count >= 10:
        print('Checkpoint appears complete.')
    elif valid_count >= 5:
        print('Checkpoint appears partial.')
    else:
        print('Checkpoint appears incomplete.')

    return 0


def main():
    parser = argparse.ArgumentParser(description='Check DCPS checkpoint contents.')
    parser.add_argument('--checkpoint', required=True, help='Path to a checkpoint file, e.g. ckpt/exp_dcps/Caltech101.pth')
    args = parser.parse_args()
    raise SystemExit(check_checkpoint(Path(args.checkpoint)))


if __name__ == '__main__':
    main()
