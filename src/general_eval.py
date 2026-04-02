import sys

if "-h" in sys.argv or "--help" in sys.argv:
    from .args import parse_arguments
    parse_arguments()
    raise SystemExit(0)

import os

import numpy as np
import torch
from CLIP.clip import clip as clip_origin
from torch import nn
from tqdm import tqdm

import custom_clip
from custom_clip.PromptCross import CustomCLIP_CPrompt
# from custom_clip.PromptHierarchical import CustomCLIP_HPrompt
# from custom_clip.promptIndependent import CustomCLIP_IPrompt
# from custom_clip.promptMutual import CustomCLIP_MPrompt
from . import datasets
from . import utils
from .args import parse_arguments
from .datasets.common import get_dataloader, maybe_dictionarize
from .models.modeling import create_image_classifier
from .models.modeling import create_zeroshot_classifier_head


def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = "features"
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = "images"
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    batched_data = enumerate(dataloader)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0

        for i, data in tqdm(batched_data):
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data["labels"].to(device)
            logits, feature = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n
    print(f"[accuracy] {top1:4f}")
    print(" ")

    metrics = {}
    metrics["top1"] = top1

    return metrics


def evaluate_fc(image_classifier, dataset):
    info = vars(args)
    old_head = image_classifier.classification_head

    for i, dataset_name in enumerate(dataset):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        image_classifier.classification_head = create_zeroshot_classifier_head(
            args, dataset=dataset_name
        )

        results = eval_single_dataset(image_classifier, dataset, args)

    image_classifier.classification_head = old_head

    return info


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
@torch.no_grad()
def zeroshot_eval(model, loader, prompts=None):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()
        # # predict
        logits, image_features,_ = model(images, prompts)

        # 将batch分割成16大小的子批次
        batch_size = images.size(0)
        sub_batch_size = 16

        all_logits = []
        all_image_features = []
        all_losses = []
        all_prob_maps = []
        all_preds = []
        all_features = []
        clip_weights = model.clip_classifier(model.prompt_learner.classnames, model.origin_model)
        # 处理每个子批次
        for j in range(0, batch_size, sub_batch_size):
            end_idx = min(j + sub_batch_size, batch_size)
            sub_images = images[j:end_idx]

            # 预测子批次
            sub_logits, sub_image_features,_ = model(sub_images, prompts)

            # 处理每个子批次的CLIP logits
            sub_loss, sub_prob_map, sub_preds, sub_features = model.get_clip_logits(sub_logits, sub_image_features)

            # 应用缓存机制到子批次
            sub_final_logits = model.cache(sub_logits, sub_features, clip_weights, sub_loss, sub_prob_map, sub_preds)

            # 收集结果
            all_logits.append(sub_final_logits)
            all_image_features.append(sub_image_features)
            all_losses.append(sub_loss)
            all_prob_maps.append(sub_prob_map)
            all_preds.append(sub_preds)
            all_features.append(sub_features)

        # 合并所有子批次的结果
        logits = torch.cat(all_logits, dim=0)
        if prompts is not None:
            logits, _ = logits

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def zeroshot_eval_origin(model, loader, dataset):
    text_features = []
    for c in dataset.classnames:
        # 针对当前类别生成多个文本描述
        texts = [template(c.replace("_", " ")) for template in dataset.templates]
        text_tokens = clip_origin.tokenize(texts).to("cuda")
        with torch.no_grad():
            text_embeds = model.encode_text(text_tokens)
            text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        # 对多个 prompt 生成的文本特征取平均作为最终的类别文本嵌入
        text_features.append(text_embeds.mean(dim=0))
    text_features = torch.stack(text_features, dim=0)  # 形状：[num_classes, feat_dim]

    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        with torch.no_grad():
            # 对批量图像进行编码
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 计算图像特征与文本特征之间的余弦相似度，并乘以放缩因子 100.0（CLIP 框架中的常用处理方式）
            logits = 100.0 * image_features @ text_features.T
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def compute_cosine_similarity_and_get_max_index(text_feature, prototype_feature):
    # first_feature is expected to be of shape (n, 512)
    # prompt_feature is expected to be of shape (11, 512)
    # Normalize the features
    # Instantiate a CosineSimilarity module
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    # Compute the cosine similarity. The result will be of shape (n, 11)
    # similarity = torch.empty(first_feature.size(0), prototype_feature.size(0))
    text_feature_mean = text_feature.mean(dim=0, keepdim=True)

    # for i in range(first_feature.size(0)):
    #     for j in range(prototype_feature.size(0)):
    similarity = cos_sim(text_feature_mean, prototype_feature)

    # Sum over the first dimension. The result will be of shape (1, 11)
    # summed_similarity = torch.mean(similarity, dim=0, keepdim=True)

    # Get the index of the maximum value in the second dimension. The result will be a scalar.
    max_index = torch.argmax(similarity).item()
    if similarity[max_index] < 0.98:
        max_index = -1
    return max_index


def eval_single_dataset_bl(model, dataset, args, RECORD):
    input_key = "images"
    image_enc = None
    with torch.no_grad():

        prompts = model.prompt_learner.tokenized_prompts.to("cuda")
        template = dataset.template

        texts = [template(c) for c in dataset.classnames]

        origin_model, _ = clip_origin.load('ViT-B/16', jit=False)
        texts = clip_origin.tokenize(texts).to("cuda")

        with torch.no_grad():
            text_feature = origin_model.encode_text(prompts)

    task_id = compute_cosine_similarity_and_get_max_index(text_feature, model.prototype_feature)

    print("the dataset id is: ", task_id)
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    if task_id != -1:
        model.select_prompt(task_id)
        texts = None
        model.eval()

        top1, top5 = zeroshot_eval(model, dataloader, texts)

    else:
        model = origin_model
        model.eval()

        top1, top5 = zeroshot_eval_origin(model, dataloader, dataset)

    print(f"Top-1 accuracy: {top1:.2f}")
    RECORD.append(top1)


def evaluate_bl(image_classifier, dataset, val_preprocess, RECORD):
    eval_single_dataset_bl(image_classifier, dataset, args, RECORD)


def evaluate(dictionary, args, fc, RECORD):
    for i in dictionary:
        model_ckpt = i[0]
        datasets_list = i[1]
        if fc:
            print(f"using checkpoint {model_ckpt}")
            model = create_image_classifier(args, setnone=True)
            utils.torch_load(model, model_ckpt)
            evaluate_fc(model, datasets)
        else:
            clip_model, _, val_preprocess = custom_clip.load(args.model, args, jit=False)
            clip_model = clip_model.to("cpu")
            for i, dataset_name in enumerate(datasets_list):
                print("Evaluating on", dataset_name)
                dataset_class = getattr(datasets, dataset_name)
                dataset = dataset_class(
                    val_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size,
                    batch_size_eval=args.batch_size_eval,
                )

                # if args.trainer == "MPrompt":
                #     model = CustomCLIP_MPrompt(args, dataset.classnames, clip_model)
                # elif args.trainer == 'IPrompt':
                #     model = CustomCLIP_IPrompt(args, dataset.classnames, clip_model)
                # elif args.trainer == 'HPrompt':
                #     model = CustomCLIP_HPrompt(args, dataset.classnames, clip_model)
                if args.trainer == "DCPS":
                    model = CustomCLIP_CPrompt(args, dataset, clip_model)

                utils.torch_load(model, model_ckpt)
                model.cuda()
                evaluate_bl(model, dataset, val_preprocess, RECORD)
    return


if __name__ == "__main__":
    args = parse_arguments()
    NAMES_ARRAY = []
    NAMES_ARRAY.append(args.eval_names)
    for NAMES in NAMES_ARRAY:
        RECORD = []
        fc = False
        # dictionary = [
        #     [f"ckpt/{NAMES}/StanfordCars.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/Food.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/MNIST.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/OxfordPet.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/Flowers.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/SUN397.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/Aircraft.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/Caltech101.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/DTD.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/EuroSAT.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/CIFAR100.pth",
        #      ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #       "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        # ]
        dictionary = [
            [f"ckpt/{NAMES}/Aircraft.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/Caltech101.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/CIFAR100.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/DTD.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/EuroSAT.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/Flowers.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/Food.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/MNIST.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/OxfordPet.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/StanfordCars.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
            [f"ckpt/{NAMES}/SUN397.pth",
             ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
              "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
        ]
        evaluate(dictionary, args, fc, RECORD)
        RECORD = np.array(RECORD)
        os.makedirs(args.output_dir, exist_ok=True)
        outfile = os.path.join(args.output_dir, f"{NAMES}.npy")
        np.save(outfile, RECORD)
        print(" ")
        print(" ")
        print(" ")
        print(f"Results saved to: {outfile}")
