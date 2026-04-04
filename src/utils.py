import os
import random

import numpy as np
import torch


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(classifier, save_path):
    save_dict = {}

    # 定义要保存的键列表
    save_keys = [
        'scale_I_pool', 'scale_T_pool',
        'proj_down_weight', 'proj_down_bias', 'proj_up_weight', 'proj_up_bias',
        'scale_vector_image', 'scale_vector',
        'ItoT_down_weight', 'ItoT_down_bias', 'ItoT_up_weight', 'ItoT_up_bias',
        'TtoI_down_weight', 'TtoI_down_bias', 'TtoI_up_weight', 'TtoI_up_bias',
        'prototype_feature', 'prompt_pool', 'text_prompt_pool', 'visual_prompt_pool', 'A_prime_pool'
    ]

    saved_keys = []
    for key in save_keys:
        if hasattr(classifier, key):
            save_dict[key] = getattr(classifier, key)
            saved_keys.append(key)

    # Make sure the save directory exists
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the dictionary
    torch.save(save_dict, save_path)

    # 打印保存状态和值
    # print("="*50)
    # print("Checkpoint 保存报告")
    for key in saved_keys:
        val = save_dict[key]
        # if hasattr(val, 'shape'):
        #     print(f"[{key}] 保存成功 | shape: {val.shape} | 前3个值: {val.flatten()[:3]}")
        # else:
        #     print(f"[{key}] 保存成功 | 值: {val}")
    for key in save_keys:
        if key not in saved_keys:
            print(f"[{key}] 模型中不存在")
    print(f"Checkpoint saved to {save_path}")
    print("="*50)


# def torch_save(classifier, save_path):
#     if os.path.dirname(save_path) != "":
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     # torch.save({"state_dict": classifier.state_dict()}, save_path)
#     if callable(classifier.state_dict):
#         state_dict = classifier.state_dict()
#     else:
#         state_dict = classifier.state_dict
#     torch.save({"state_dict": state_dict}, save_path)
#     print("Checkpoint saved to", save_path)
# def torch_load(classifier, save_path, device=None):
#     print("当前工作目录:", os.getcwd())
#     print(f"文件存在: {os.path.exists(save_path)}")
#     absolute_path = os.path.abspath(save_path)
#     print("绝对路径:", absolute_path)
#     checkpoint = torch.load(save_path, weights_only=False)
#
#
#     # 处理特定类型的属性加载
#     if 'visual_proj_pool' in checkpoint and hasattr(classifier, 'visual_proj_pool'):
#         classifier.visual_proj_pool.load_state_dict(checkpoint['visual_proj_pool'])
#
#     # 加载直接属性
#     for key in ['visual_align_pool', 'text_align_pool', 'prototype_feature',
#                 'text_prompt_pool', 'visual_prompt_pool']:
#         if key in checkpoint and hasattr(classifier, key):
#             setattr(classifier, key, checkpoint[key])
#
#     if device is not None:
#         classifier = classifier.to(device)
#     return classifier

def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path, weights_only=False)

    print("="*50)
    print("Checkpoint 加载报告")

    # ===== 3. 加载 buffer（使用 register_buffer） =====
    buffer_keys = [
        'prototype_feature', 'prompt_pool', 'text_prompt_pool', 'visual_prompt_pool', 'A_prime_pool',
        'scale_I_pool', 'scale_T_pool',
        'proj_down_weight', 'proj_down_bias', 'proj_up_weight', 'proj_up_bias',
        'scale_vector_image', 'scale_vector',
        'ItoT_down_weight', 'ItoT_down_bias', 'ItoT_up_weight', 'ItoT_up_bias',
        'TtoI_down_weight', 'TtoI_down_bias', 'TtoI_up_weight', 'TtoI_up_bias'
    ]
    for key in buffer_keys:
        if key in checkpoint:
            classifier.register_buffer(key, checkpoint[key])
            val = checkpoint[key]
            # if hasattr(val, 'shape'):
            #     print(f"[{key}] 加载成功 | shape: {val.shape} | 前3个值: {val.flatten()[:3]}")
            # else:
            #     print(f"[{key}] 加载成功 | 值: {val}")
        else:
            print(f"[{key}] checkpoint中不存在")

    print("="*50)

    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
