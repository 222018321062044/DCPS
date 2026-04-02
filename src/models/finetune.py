import os

from CLIP.clip import clip

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .evaluation import evaluate
from .. import datasets, templates, utils


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def finetune(args):
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
    if args.load is not None:
        utils.torch_load(model, args.load)

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
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
    if args.train_mode == "text":
        print("[Training mode] Text Encoder")
        visual_params_name = [k for k, v in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif args.train_mode == "image":
        print("[Training mode] Image Encoder")
        params = model.visual.parameters()

    elif args.train_mode == "prompt":
        print("[Training mode] Prompt Tuning")
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Add your own trainable parameters
        # params = add_trainable_parameters(model)

    else:
        assert args.train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if eval_iterations is not None and iteration % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess)

        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data

        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())

    # Saving model
    if args.save is not None:
        to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
