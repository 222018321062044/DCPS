import os
import random
from collections import OrderedDict

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


CHECKPOINT_EXPORT_KEYS = (
    "prototype_feature",
    "prompt_pool",
    "text_prompt_pool",
    "visual_prompt_pool",
    "proj_down_weight",
    "proj_down_bias",
    "proj_up_weight",
    "proj_up_bias",
    "ItoT_down_weight",
    "ItoT_down_bias",
    "ItoT_up_weight",
    "ItoT_up_bias",
    "TtoI_down_weight",
    "TtoI_down_bias",
    "TtoI_up_weight",
    "TtoI_up_bias",
    "scale_vector_image",
    "scale_vector",
    "A_prime_pool",
)


def _clone_tensor(value):
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()
    elif torch.is_tensor(value):
        value = value.detach()
    else:
        raise TypeError(f"Expected tensor-like value, got {type(value).__name__}")
    return value.cpu().clone()


def _cpu_state_dict(module):
    state_dict = OrderedDict()
    for key, value in module.state_dict().items():
        state_dict[key] = _clone_tensor(value)
    return state_dict


def _infer_pool_width(pool, fallback_tensor=None):
    if fallback_tensor is not None and torch.is_tensor(fallback_tensor):
        return int(fallback_tensor.numel())
    for row in pool:
        if torch.is_tensor(row) or isinstance(row, torch.nn.Parameter):
            return int(row.numel())
    return 0


def _materialize_vector_pool(pool, fallback_tensor=None):
    if torch.is_tensor(pool):
        return _clone_tensor(pool)

    width = _infer_pool_width(pool, fallback_tensor=fallback_tensor)
    rows = []
    for row in pool:
        if torch.is_tensor(row) or isinstance(row, torch.nn.Parameter):
            rows.append(_clone_tensor(row).reshape(-1))
        else:
            rows.append(torch.zeros(width, dtype=torch.float32))

    if not rows:
        return torch.empty(0, width, dtype=torch.float32)
    return torch.stack(rows, dim=0)


def torch_save(classifier, save_path):
    if hasattr(classifier, "state_dict") and callable(classifier.state_dict):
        state_dict = _cpu_state_dict(classifier)
        save_dict = {
            "checkpoint_version": 2,
            "state_dict": state_dict,
        }
    else:
        if not isinstance(classifier, dict):
            raise TypeError(
                "torch_save expects a module with state_dict() or a plain dict checkpoint."
            )
        save_dict = {
            "checkpoint_version": 2,
            "state_dict": classifier,
        }
        state_dict = classifier

    for key in CHECKPOINT_EXPORT_KEYS:
        if key in state_dict:
            save_dict[key] = state_dict[key]

    if hasattr(classifier, "scale_I_pool"):
        fallback = None
        if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_I"):
            fallback = classifier.prompt_learner.scale_I
        save_dict["scale_I_pool"] = _materialize_vector_pool(
            classifier.scale_I_pool,
            fallback_tensor=fallback,
        )

    if hasattr(classifier, "scale_T_pool"):
        fallback = None
        if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_T"):
            fallback = classifier.prompt_learner.scale_T
        save_dict["scale_T_pool"] = _materialize_vector_pool(
            classifier.scale_T_pool,
            fallback_tensor=fallback,
        )

    if hasattr(classifier, "visual_proj_pool"):
        save_dict["visual_proj_pool"] = {
            key: _clone_tensor(value)
            for key, value in classifier.visual_proj_pool.state_dict().items()
        }

    # Make sure the save directory exists
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the dictionary
    torch.save(save_dict, save_path)
    print("Checkpoint saved to", save_path)


def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}

    model_state = classifier.state_dict()
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        load_state = OrderedDict(checkpoint["state_dict"])
    else:
        load_state = OrderedDict(
            (key, value)
            for key, value in checkpoint.items()
            if key in model_state and torch.is_tensor(value)
        )

    for key, value in checkpoint.items():
        if key in model_state and torch.is_tensor(value):
            load_state[key] = value

    missing_keys, unexpected_keys = classifier.load_state_dict(load_state, strict=False)

    if "scale_I_pool" in checkpoint and hasattr(classifier, "scale_I_pool"):
        scale_i_pool = checkpoint["scale_I_pool"]
        if not torch.is_tensor(scale_i_pool):
            fallback = None
            if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_I"):
                fallback = classifier.prompt_learner.scale_I
            scale_i_pool = _materialize_vector_pool(
                scale_i_pool,
                fallback_tensor=fallback,
            )
        if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_I"):
            scale_i_pool = scale_i_pool.to(classifier.prompt_learner.scale_I.device)
        setattr(classifier, "scale_I_pool", scale_i_pool)

    if "scale_T_pool" in checkpoint and hasattr(classifier, "scale_T_pool"):
        scale_t_pool = checkpoint["scale_T_pool"]
        if not torch.is_tensor(scale_t_pool):
            fallback = None
            if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_T"):
                fallback = classifier.prompt_learner.scale_T
            scale_t_pool = _materialize_vector_pool(
                scale_t_pool,
                fallback_tensor=fallback,
            )
        if hasattr(classifier, "prompt_learner") and hasattr(classifier.prompt_learner, "scale_T"):
            scale_t_pool = scale_t_pool.to(classifier.prompt_learner.scale_T.device)
        setattr(classifier, "scale_T_pool", scale_t_pool)

    for key in unexpected_keys:
        if key in {"state_dict", "checkpoint_version", "scale_I_pool", "scale_T_pool"}:
            continue
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
