import os

import torch
from tqdm import tqdm
from custom_clip import custom_clip as clip
from custom_clip.PromptCross import CustomCLIP_CPrompt
# from custom_clip.PromptHierarchical import CustomCLIP_HPrompt
# from custom_clip.promptIndependent import CustomCLIP_IPrompt
# from custom_clip.promptMutual import CustomCLIP_MPrompt
from .evaluation import evaluate
from .. import datasets, templates, utils

def prompt_tune(args):
    TaskID_dict = {
        "Aircraft": 0,
        "Caltech101": 1,
        "CIFAR100": 2,
        "DTD": 3,
        "EuroSAT": 4,
        "Flowers": 5,
        "Food": 6,
        "MNIST": 7,
        "OxfordPet": 8,
        "StanfordCars": 9,
        "SUN397": 10
    }
    task_id = TaskID_dict[args.train_dataset]
    clip_model, train_preprocess, val_preprocess = clip.load(args.model, args, jit=False)
    clip_model = clip_model.to("cpu")
    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        few_shot=args.few_shot,
    )
    # if args.trainer == "MPrompt":
    #     model = CustomCLIP_MPrompt(args, dataset.classnames, clip_model)
    # elif args.trainer == 'IPrompt':
    #     model = CustomCLIP_IPrompt(args, dataset.classnames, clip_model)
    # elif args.trainer == 'HPrompt':
    #     model = CustomCLIP_HPrompt(args, dataset.classnames, clip_model)
    if args.trainer == "DCPS":
        model = CustomCLIP_CPrompt(args, dataset, clip_model)

    if args.load is not None:
        utils.torch_load(model, args.load)
    model = model.cuda()
    model.update_prototype_feature(task_id=task_id)

    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params

    assert args.train_mode == "prompt"
    print("[Training mode] Prompt")
    # include_params_name = "VPT"
    name_to_update = "prompt_learner"
    # for name, param in model.named_parameters():
    #     if "prompt_learner" not in name:
    #         param.requires_grad_(False)
    #     else:
    #         print(f"Trainable parameter: {name}")
    for name, param in model.named_parameters():
        # 定义需要更新梯度的模块名称关键词
        update_keywords = [name_to_update,"VPT","scale_vector"]
        # update_keywords = []
        # 检查当前参数是否包含任何一个关键词
        should_update = any(keyword in name for keyword in update_keywords)

        if should_update:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    if args.trainer == "HPrompt":
        meta_net_params = list(model.prompt_learner.meta_net.parameters())  # parameters of the meta_net
        meta_net_param_ids = set(id(p) for p in model.prompt_learner.meta_net.parameters())

        base_params = [p for p in model.parameters() if id(p) not in meta_net_param_ids]
        params = [
            {'params': base_params},
            {'params': meta_net_params, 'lr': args.meta_net_lr}
        ]
    else:
        params = model.parameters()

    def count_parameters(model):
        # 打印模型参数详细信息
        print("=" * 80)
        print("Model Parameters Details:")
        print("=" * 80)

        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                trainable_status = "Trainable"
                print(f"{name:<60} | Shape: {str(param.shape):<20} | Params: {num_params:>9,} | Status: {trainable_status}")
            else:
                trainable_status = "Frozen"

        print("=" * 80)
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        print(f'Total Parameters: {total_params:,}')
        print(f'Trainable Parameters: {trainable_params:,}')
        print(f'Trainable Parameters Percentage: {percentage:.2f}%')
        print("=" * 80)

        return total_params, trainable_params, percentage

    # 调用示例
    total, trainable, percentage = count_parameters(model)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.wd,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
        )

    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device

    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)

    data_iter = iter(dataset.train_loader)
    for iteration in tqdm(range(1, total_iterations + 1)):

        # training
        # if iteration % num_batches == 0:
        #     data_iter = iter(dataset.train_loader)

        # prepare model
        model=model.to("cuda")
        model.train()

        scheduler(iteration)

        # prepare data
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()
        # ce loss
        # -- get text embedding --
        # if args.train_mode != "text":
        #     embeddings = model(None, texts)
        #     embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        loss = model(images, labels)

        # out = out / out.norm(dim=-1, keepdim=True)
        #
        # # -- cross entropy loss --
        # logits_per_image = logit_scale.exp() * out @ embeddings.t()
        # loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if eval_iterations is not None and iteration % (eval_iterations) == 0:
            model.update_pool(task_id, args.prompt_depth_text, args.prompt_depth_vision)
            evaluate(model, args, val_preprocess, iteration=iteration)

        if iteration % loss_interval == 0:
            print("Loss:", loss.item())

    model.update_pool(task_id, args.prompt_depth_text, args.prompt_depth_vision)

    # Saving model
    if args.save is not None:
        to_save_model = model
        # dict1=model.state_dict()
        # to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
    return model
