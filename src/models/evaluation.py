import torch
import torch.nn as nn
from CLIP.clip import clip as clip_origin
from tqdm import tqdm

from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_eval(
    model,
    loader,
    prompts,
    iteration=0,
    dataset=None,
    args=None,
):
    top1, top5, n = 0.0, 0.0, 0.0
    clip_weights = model.clip_classifier(
        model.prompt_learner.classnames,
        model.origin_model,
    )

    for data in tqdm(loader):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        logits, image_features, _ = model(images, prompts)
        loss, prob_map, preds, features = model.get_clip_logits(logits, image_features)
        logits = model.cache(logits, features, clip_weights, loss, prob_map, preds)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    return (top1 / n) * 100, (top5 / n) * 100


@torch.no_grad()
def zeroshot_eval_origin(model, loader, dataset):
    text_features = []
    for class_name in dataset.classnames:
        texts = [template(class_name.replace("_", " ")) for template in dataset.templates]
        text_tokens = clip_origin.tokenize(texts).to("cuda")
        text_embeds = model.encode_text(text_tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        text_features.append(text_embeds.mean(dim=0))
    text_features = torch.stack(text_features, dim=0)

    top1, top5, n = 0.0, 0.0, 0.0
    for data in tqdm(loader):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    return (top1 / n) * 100, (top5 / n) * 100


def compute_cosine_similarity_and_get_max_index(text_feature, prototype_feature):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    text_feature_mean = text_feature.mean(dim=0, keepdim=True)
    similarity = cos_sim(text_feature_mean, prototype_feature)
    max_index = torch.argmax(similarity).item()
    if similarity[max_index] < 0.97:
        return -1
    return max_index


def eval_single_dataset(image_classifier, dataset, args, iteration=0, val_preprocess=None):
    with torch.no_grad():
        prompts = image_classifier.prompt_learner.tokenized_prompts.to("cuda")
        origin_model, _ = clip_origin.load("ViT-B/16", jit=False)
        text_feature = origin_model.encode_text(prompts)

    task_id = compute_cosine_similarity_and_get_max_index(
        text_feature,
        image_classifier.prototype_feature,
    )
    print("the dataset id is: ", task_id)

    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    if task_id != -1:
        image_classifier.select_prompt(task_id)
        image_classifier.eval()
        top1, top5 = zeroshot_eval(image_classifier, dataloader, prompts=None)
    else:
        origin_model.eval()
        top1, top5 = zeroshot_eval_origin(origin_model, dataloader, dataset)

    print(f"Top-1 accuracy: {top1:.2f}")
    return top1, top5


def evaluate(image_classifier, args, val_preprocess, iteration=0):
    if args.eval_datasets is None:
        return

    for dataset_name in args.eval_datasets:
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset(image_classifier, dataset, args, iteration=iteration)
