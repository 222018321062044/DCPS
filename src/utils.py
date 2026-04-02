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
    # Temporarily remove dataset to avoid pickle errors (contains lambda functions)
    dataset_backup = None
    if hasattr(classifier, 'dataset'):
        dataset_backup = classifier.dataset
        classifier.dataset = None

    # Create a dictionary to hold the parts of the model to be saved
    save_dict = {}

    # ★ 保存完整的state_dict（包括prompt_learner的A_prime等参数）
    # Check if classifier is a model with state_dict method or already a state_dict
    if hasattr(classifier, 'state_dict') and callable(classifier.state_dict):
        save_dict['state_dict'] = classifier.state_dict()
    else:
        # classifier is already a state_dict (OrderedDict)
        save_dict = classifier if isinstance(classifier, dict) else {'state_dict': classifier}

    # Restore dataset before saving extra attributes
    if dataset_backup is not None:
        classifier.dataset = dataset_backup

    # Save the state_dict of the visual_proj_pool ModuleList
    if hasattr(classifier, 'visual_proj_pool'):
        save_dict['visual_proj_pool'] = classifier.visual_proj_pool.state_dict()

    if hasattr(classifier, 'visual_align_pool'):
        save_dict['visual_align_pool'] = classifier.visual_align_pool
    if hasattr(classifier, 'text_align_pool'):
        save_dict['text_align_pool'] = classifier.text_align_pool
    if hasattr(classifier, 'scale_I_pool'):
        save_dict['scale_I_pool'] = classifier.scale_I_pool
    if hasattr(classifier, 'scale_T_pool'):
        save_dict['scale_T_pool'] = classifier.scale_T_pool
    if hasattr(classifier, 'proj_down_weight'):
        save_dict['proj_down_weight'] = classifier.proj_down_weight
    if hasattr(classifier, 'proj_up_weight'):
        save_dict['proj_up_weight'] = classifier.proj_up_weight
    if hasattr(classifier, 'proj_down_bias'):
        save_dict['proj_down_bias'] = classifier.proj_down_bias
    if hasattr(classifier, 'proj_up_bias'):
        save_dict['proj_up_bias'] = classifier.proj_up_bias
    if hasattr(classifier, 'scale_vector_image'):
        save_dict['scale_vector_image'] = classifier.scale_vector_image
    if hasattr(classifier, 'scale_vector'):
        save_dict['scale_vector'] = classifier.scale_vector
    if hasattr(classifier, 'ItoT_down_weight'):
        save_dict['ItoT_down_weight'] = classifier.ItoT_down_weight
    if hasattr(classifier, 'ItoT_up_weight'):
        save_dict['ItoT_up_weight'] = classifier.ItoT_up_weight
    if hasattr(classifier, 'ItoT_down_bias'):
        save_dict['ItoT_down_bias'] = classifier.ItoT_down_bias
    if hasattr(classifier, 'ItoT_up_bias'):
        save_dict['ItoT_up_bias'] = classifier.ItoT_up_bias
    if hasattr(classifier, 'TtoI_down_weight'):
        save_dict['TtoI_down_weight'] = classifier.TtoI_down_weight
    if hasattr(classifier, 'TtoI_up_weight'):
        save_dict['TtoI_up_weight'] = classifier.TtoI_up_weight
    if hasattr(classifier, 'TtoI_down_bias'):
        save_dict['TtoI_down_bias'] = classifier.TtoI_down_bias
    if hasattr(classifier, 'TtoI_up_bias'):
        save_dict['TtoI_up_bias'] = classifier.TtoI_up_bias
    # Save the prototype_feature buffer
    save_dict['prototype_feature'] = classifier.prototype_feature
    save_dict['prompt_pool'] = classifier.prompt_pool
    # Save the text_prompt_pool buffer
    save_dict['text_prompt_pool'] = classifier.text_prompt_pool

    # Save the visual_prompt_pool buffer
    if hasattr(classifier, 'visual_prompt_pool'):
        save_dict['visual_prompt_pool'] = classifier.visual_prompt_pool

    # Make sure the save directory exists
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the dictionary
    torch.save(save_dict, save_path)
    print("Checkpoint saved to", save_path)


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

    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint, strict=False
    )
    for key in unexpected_keys:
        if hasattr(classifier, key):
            setattr(classifier, key, checkpoint[key])

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
