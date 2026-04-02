import math
import operator

import numpy as np
import torch
import torch.nn as nn
from numpy.matlib import empty
from sympy.physics.paulialgebra import delta
from torch.nn.functional import embedding

from CLIP.clip import clip as clip_origin

from torch.nn import functional as F
from custom_clip import custom_clip as clip
from custom_clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    eps = 1e-10  # small constant to avoid log(0)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0] + eps)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


# class LightweightPromptGenerator(nn.Module):
#     def __init__(self, input_dim, prompt_dim, num_layers=12, hidden_dim=256):
#         super().__init__()
#         self.input_dim = input_dim
#         self.prompt_dim = prompt_dim
#         self.num_layers = num_layers
#
#         # 使用小型MLP生成提示
#         self.generator_visual = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_layers * prompt_dim * 2),  # 同时生成视觉和文本提示
#             nn.Tanh()  # 限制输出范围
#         )
#         self.generator_text = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_layers * 768 * 2),  # 同时生成视觉和文本提示
#             nn.Tanh()  # 限制输出范围
#         )
#         # 可学习的基准提示
#         self.base_visual_prompts = nn.Parameter(torch.randn(num_layers, 2, prompt_dim) * 0.02)
#         self.base_text_prompts = nn.Parameter(torch.randn(num_layers, 2, 768) * 0.02)
#         self.scale_I = nn.Parameter(torch.randn(prompt_dim) * 0.02)
#         self.scale_T = nn.Parameter(torch.randn(768) * 0.02)
#
#     def forward(self, image_features):
#         global_feat = image_features.mean(dim=0)
#         prompt_delta_visual = self.generator_visual(global_feat)
#         prompt_delta_visual = prompt_delta_visual.view(self.num_layers, 2, self.prompt_dim)
#         prompt_delta_text = self.generator_text(global_feat)
#         prompt_delta_text = prompt_delta_text.view(self.num_layers, 2, 768)
#
#         # 添加基准提示
#         visual_prompts = self.base_visual_prompts + prompt_delta_visual * self.scale_I
#         text_prompts = self.base_text_prompts + prompt_delta_text * self.scale_T
#
#         return visual_prompts, text_prompts
class MLPAligner(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 bottleneck_dim=256,
                 dropout_rate=0.02,
                 activation='gelu'):
        super(MLPAligner, self).__init__()

        # 选择激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 从输入维度压缩到瓶颈维度
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)

        # 从瓶颈维度扩展回原始维度
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)

        # Dropout层用于正则化
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入张量形状: [batch_size, seq_len, input_dim]

        # 向下映射到瓶颈维度
        h = self.down_proj(x)
        h = self.activation(h)

        # 应用dropout
        h = self.dropout(h)
        # 向上映射回原始维度
        output = self.up_proj(h)

        return output
class PromptLearner(nn.Module):
    def __init__(self, args, dataset, clip_model):
        super().__init__()
        self.prompts_Image = torch.empty(2, 512)
        classnames = dataset.classnames
        n_cls = len(classnames)
        dtype = clip_model.dtype
        templates = getattr(dataset, 'templates', [lambda c: f"a photo of a {c}."])
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [" ".join(["X"] * 2) + " " + name + "." for name in classnames]
        tokenized_prompts_outer = torch.cat([clip.tokenize(p) for p in prompts])
        all_prompts = []
        num_templates = min(4, len(templates))
        tokenized_prompts_outer = tokenized_prompts_outer.repeat_interleave(num_templates, dim=0)
        tokenized_prompts_outer = tokenized_prompts_outer.cuda()
        for classname in classnames:
            for template in templates[:min(4, len(templates))]:

                all_prompts.append(template(classname))

        tokenized_prompts = clip.tokenize(all_prompts).cuda()

        with torch.no_grad():
            token_embedding = clip_model.token_embedding.cuda()
            self.embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding = token_embedding(tokenized_prompts_outer).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.n_templates = min(4, len(templates))
        self.n_cls = n_cls
        self.classnames = classnames
        self.templates = templates
        #Low-rank decomposition parameters
        # bottleneck_dim = 257
        # self.bottleneck_dim = bottleneck_dim
        # # A' and B' matrices (Eq.6 in paper)
        self.A_prime = nn.Parameter(torch.empty(2, 512, dtype=dtype))
        # self.B_prime = nn.Parameter(torch.empty(bottleneck_dim-1, 512, dtype=dtype))

        # # Initialize parameters
        nn.init.normal_(self.A_prime, std=0.02)

        self.scale_I = nn.Parameter(torch.randn(512) * 0.02)
        self.scale_T = nn.Parameter(torch.randn(512) * 0.02)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + 2:, :])  # CLS, EOS
        self.projection_images = MLPAligner(512,512)
        self.TtoI = MLPAligner(512, 768)
        self.ItoT = MLPAligner(768, 512)
    def forward(self, image_features, template_embeddings):
        # self.prompts_Image = torch.matmul(self.A_prime, self.B_prime)
        base_prompts = self.A_prime
        delta = torch.zeros_like(base_prompts)
        image_features = self.projection_images(image_features)
        delta += (self.scale_I * image_features).unsqueeze(0)
        ctx = base_prompts + delta
        ctx = ctx.unsqueeze(0).expand(self.n_cls * self.n_templates, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts + self.scale_T * template_embeddings
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, VPTs=None, n_cls=None, n_templates=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x, VPTs)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if n_cls is not None and n_templates is not None:
            x = x.view(n_cls, n_templates, -1).mean(dim=1)

        return x
class CustomCLIP_CPrompt(nn.Module):
    def __init__(self, args, dataset, clip_model):
        super().__init__()
        self.text_VPTs = []
        self.visual_VPTs= []
        self.origin_model, _ = clip_origin.load('ViT-B/16', jit=False)
        self.prompt_learner = PromptLearner(args, dataset, clip_model)
        text_prompt_pool = torch.empty(11, args.prompt_depth_text, args.n_ctx_text, 512, device=device)
        prompt_pool = torch.empty(11, 2, 512, device=device)
        visual_prompt_pool = torch.empty(11, args.prompt_depth_vision, args.n_ctx_vision, 768, device=device)
        self.scale_T_pool = [[None] * 512 for _ in range(11)]
        self.scale_I_pool = [[None] * 512 for _ in range(11)]
        self.hard_sample_ratio = args.hard_sample_ratio
        self.threshold_percentile = args.threshold_percentile
        self.hard_loss_weight = args.hard_loss_weight
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.image_encoder_original = self.origin_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.dtype2 = self.origin_model.dtype
        prototype_feature = torch.empty(11, 512, device=device)
        self.prompt_learner = self.prompt_learner.to(device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts = self.tokenized_prompts.cuda()
        self.dataset = dataset
        self.args = args

        self.register_buffer("prototype_feature", prototype_feature)
        self.register_buffer("text_prompt_pool", text_prompt_pool)
        self.register_buffer("visual_prompt_pool", visual_prompt_pool)
        self.register_buffer("prompt_pool", prompt_pool)
        self.register_buffer("proj_down_weight", torch.empty(11, 256, 512, device=device))
        self.register_buffer("proj_down_bias", torch.empty(11, 256, device=device))
        self.register_buffer("proj_up_weight", torch.empty(11, 512, 256, device=device))
        self.register_buffer("proj_up_bias", torch.empty(11, 512, device=device))
        self.register_buffer("ItoT_down_weight", torch.empty(11, 256, 768, device=device))
        self.register_buffer("ItoT_down_bias", torch.empty(11, 256, device=device))
        self.register_buffer("ItoT_up_weight", torch.empty(11, 512, 256, device=device))
        self.register_buffer("ItoT_up_bias", torch.empty(11, 512, device=device))
        self.register_buffer("TtoI_down_weight", torch.empty(11, 256, 512, device=device))
        self.register_buffer("TtoI_down_bias", torch.empty(11, 256, device=device))
        self.register_buffer("TtoI_up_weight", torch.empty(11, 768, 256, device=device))
        self.register_buffer("TtoI_up_bias", torch.empty(11, 768, device=device))
        self.register_buffer("scale_vector_image",torch.empty(11, args.prompt_depth_vision, 512, device=device))
        self.register_buffer("scale_vector", torch.empty(11, args.prompt_depth_vision, 768, device=device))
        # Pool for A_prime parameter (core learnable prompt parameter)
        self.register_buffer("A_prime_pool", torch.empty(11, 2, 512, device=device))
        # Initialize cache as instance variables for persistent storage
        self.pos_cache = {}
        self.neg_cache = {}
    def forward(self, image, label=None):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        self.prompts = self.prompt_learner.embedding.cuda()

        # 12个[2,768]的提示
        self.visual_VPTs = [layer.VPT_shallow for layer in self.image_encoder.transformer.resblocks if layer.add_prompt]

        # 12个[2,512]的提示
        self.text_VPTs = [layer.VPT_shallow for layer in self.text_encoder.transformer.resblocks if layer.add_prompt]
        text_context = []
        visual_context = []
        for i in range(12):
            context_t = self.text_VPTs[i]
            context_t = self.prompt_learner.TtoI(context_t)
            context_i = self.visual_VPTs[i]
            context_i = self.prompt_learner.ItoT(context_i)
            text_context.append(context_t)
            visual_context.append(context_i)
        image_features = self.image_encoder(image.type(self.dtype), text_context)
        # image_features = self.image_encoder(image.type(self.dtype),self.text_VPTs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        n_cls = self.prompt_learner.n_cls
        n_templates = self.prompt_learner.n_templates
        self.prompts_images = self.prompt_learner(image_features.mean(dim=0), self.prompts)
        # text_features = self.text_encoder(
        #     self.prompts,
        #     self.tokenized_prompts,
        #     visual_context,
        #     n_cls,
        #     n_templates
        # )
        text_features = self.text_encoder(
            prompts=self.prompts_images,
            tokenized_prompts=self.tokenized_prompts,
            VPTs=visual_context,
            n_cls=n_cls,
            n_templates=n_templates
        )
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ self.text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits, image_features, self.text_features
    def get_clip_logits(self, clip_logits, image_features):
        batch_entropy = softmax_entropy(clip_logits)
        confidence_threshold = 0.8
        high_confidence_mask = (clip_logits.softmax(1).max(1)[0] > confidence_threshold)
        if high_confidence_mask.any():
            selected_idx = torch.where(high_confidence_mask)[0]
            selected_count = min(10, len(selected_idx))
            selected_idx = selected_idx[:selected_count]
        else:
            # 如果没有高置信度样本，选择最不确定的1个
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=True)[:1]
        # selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        # print(selected_idx)
        output = clip_logits[selected_idx]
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output.mean(0).unsqueeze(0)
        output = clip_logits
        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        return loss, prob_map, pred, image_features
    def clip_classifier(self, classnames, clip_model):
        with torch.no_grad():
            clip_weights = []
            templates = getattr(self.dataset, 'templates', [lambda c: f"a photo of a {c}."])
            for i, classname in enumerate(classnames):
                # Tokenize the prompts
                texts = [template(classname) for template in templates[:min(4, len(templates))]]
                texts = clip.tokenize(texts).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        return clip_weights
    def compute_cache_logits(self, image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
        """Compute logits using positive/negative cache."""
        with torch.no_grad():
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)

            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = (
                ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
                    torch.int8)).cuda().half()
            else:
                cache_values = (
                    F.one_hot(torch.Tensor(cache_values).to(torch.int64),
                              num_classes=clip_weights.size(1))).cuda().half()

            affinity = image_features @ cache_keys
            affinity = affinity.type(self.dtype)
            cache_values = cache_values.type(self.dtype)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            return alpha * cache_logits
    # 修改cache函数调用
    def cache(self, clip_logits, image_features, clip_weights, loss, prob_map, pred,
             pos_capacity=None, neg_capacity=None, pos_alpha=None, pos_beta=None,
             neg_alpha=None, neg_beta=None, tau_l=None, tau_u=None):
        """
        缓存机制，支持可选的超参数

        Args:
            pos_capacity: 正缓存容量，默认3
            neg_capacity: 负缓存容量，默认2
            pos_alpha: 正缓存alpha，默认根据数据集设置
            pos_beta: 正缓存beta，默认根据数据集设置
            neg_alpha: 负缓存alpha，默认0.117
            neg_beta: 负缓存beta，默认1.0
            tau_l: 熵阈值下限，默认0.2
            tau_u: 熵阈值上限，默认0.5
        """
        pos_cache, neg_cache = {}, {}
        pos_enabled = True
        neg_enabled = True
        dataset_name = self.args.train_dataset

        # 默认正缓存参数
        default_pos_params = {
            'Aircraft': {'alpha': 2.0, 'beta': 2.0},
            'Caltech101': {'alpha': 5.0, 'beta': 5.0},
            'CIFAR100': {'alpha': 5.0, 'beta': 5.0},
            'DTD': {'alpha': 2.0, 'beta': 3.0},
            'EuroSAT': {'alpha': 4.0, 'beta': 8.0},
            'Flowers': {'alpha': 1.0, 'beta': 5.0},
            'Food': {'alpha': 1.0, 'beta': 1.0},
            'MNIST': {'alpha': 1.0, 'beta': 1.0},
            'OxfordPet': {'alpha': 2.0, 'beta': 7.0},
            'StanfordCars': {'alpha': 1.0, 'beta': 7.0},
            'SUN397': {'alpha': 2.0, 'beta': 3.0},
        }
        pos_shot_capacity = pos_capacity if pos_capacity is not None else 3
        pos_alpha_final = pos_alpha if pos_alpha is not None else (default_pos_params.get(dataset_name, {}).get('alpha', 4.0))
        pos_beta_final = pos_beta if pos_beta is not None else (default_pos_params.get(dataset_name, {}).get('beta', 8.0))

        # 默认负缓存参数
        neg_shot_capacity = neg_capacity if neg_capacity is not None else 2
        neg_alpha_final = neg_alpha if neg_alpha is not None else 0.117
        neg_beta_final = neg_beta if neg_beta is not None else 1.0
        tau_l_final = tau_l if tau_l is not None else 0.2
        tau_u_final = tau_u if tau_u is not None else 0.5

        prop_entropy = get_entropy(loss, clip_weights)
        if pos_enabled:
            update_cache(pos_cache, pred, [image_features, loss], pos_shot_capacity)

        if neg_enabled and tau_l_final < prop_entropy < tau_u_final:
            update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_shot_capacity, True)

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            final_logits += self.compute_cache_logits(image_features, pos_cache, pos_alpha_final, pos_beta_final,
                                                 clip_weights)
        if neg_enabled and neg_cache:
            final_logits -= self.compute_cache_logits(image_features, neg_cache, neg_alpha_final, neg_beta_final,
                                                 clip_weights, (0.03, 1.0))

        return final_logits
    def update_pool(self, task_id, prompt_depth_text, prompt_depth_vision):
        with torch.no_grad():
            self.prompt_pool[task_id] = self.prompt_learner.prompts_Image
            self.scale_I_pool[task_id] = self.prompt_learner.scale_I
            self.scale_T_pool[task_id] = self.prompt_learner.scale_T
            self.TtoI_down_weight[task_id] = self.prompt_learner.TtoI.down_proj.weight.data
            self.TtoI_down_bias[task_id] = self.prompt_learner.TtoI.down_proj.bias.data
            self.TtoI_up_weight[task_id] = self.prompt_learner.TtoI.up_proj.weight.data
            self.TtoI_up_bias[task_id] = self.prompt_learner.TtoI.up_proj.bias.data
            self.ItoT_down_weight[task_id] = self.prompt_learner.ItoT.down_proj.weight.data
            self.ItoT_down_bias[task_id] = self.prompt_learner.ItoT.down_proj.bias.data
            self.ItoT_up_weight[task_id] = self.prompt_learner.ItoT.up_proj.weight.data
            self.ItoT_up_bias[task_id] = self.prompt_learner.ItoT.up_proj.bias.data
            self.proj_down_weight[task_id] = self.prompt_learner.projection_images.down_proj.weight.data
            self.proj_down_bias[task_id] = self.prompt_learner.projection_images.down_proj.bias.data
            self.proj_up_weight[task_id] = self.prompt_learner.projection_images.up_proj.weight.data
            self.proj_up_bias[task_id] = self.prompt_learner.projection_images.up_proj.bias.data

            # for i in range(prompt_depth_text):
            #     self.text_prompt_pool[task_id][i] = self.text_encoder.transformer.resblocks[i].VPT_shallow
            #     self.scale_vector_image[task_id][i] = self.text_encoder.transformer.resblocks[i].scale_vector_image
            #
            # for i in range(prompt_depth_vision):
            #     self.visual_prompt_pool[task_id][i] = self.image_encoder.transformer.resblocks[i].VPT_shallow
            #     self.scale_vector[task_id][i] = self.image_encoder.transformer.resblocks[i].scale_vector

    def update_prototype_feature(self, task_id=None):

        if task_id is not None:
            with torch.no_grad():
                prompts = self.prompt_learner.tokenized_prompts
                prompts = prompts.to("cuda")
                text_features = self.origin_model.encode_text(prompts)

            temp_param = nn.Parameter(text_features.mean(dim=0, keepdim=True))
            temp_param = F.normalize(temp_param, dim=1)
            self.prototype_feature[task_id] = temp_param
    def select_prompt(self, task_id):
        prompt_depth_text = self.text_prompt_pool.shape[1]
        prompt_depth_vision = self.visual_prompt_pool.shape[1]
        self.prompt_learner.prompts_Image.data = self.prompt_pool[task_id]
        self.prompt_learner.scale_I.data = self.scale_I_pool[task_id]
        self.prompt_learner.scale_T.data = self.scale_T_pool[task_id]
        self.prompt_learner.projection_images.down_proj.weight.data = self.proj_down_weight[task_id]
        self.prompt_learner.projection_images.down_proj.bias.data = self.proj_down_bias[task_id]
        self.prompt_learner.projection_images.up_proj.weight.data = self.proj_up_weight[task_id]
        self.prompt_learner.projection_images.up_proj.bias.data = self.proj_up_bias[task_id]
        self.prompt_learner.ItoT.down_proj.weight.data = self.ItoT_down_weight[task_id]
        self.prompt_learner.ItoT.down_proj.bias.data = self.ItoT_down_bias[task_id]
        self.prompt_learner.ItoT.up_proj.weight.data = self.ItoT_up_weight[task_id]
        self.prompt_learner.ItoT.up_proj.bias.data = self.ItoT_up_bias[task_id]
        self.prompt_learner.TtoI.down_proj.weight.data = self.TtoI_down_weight[task_id]
        self.prompt_learner.TtoI.down_proj.bias.data = self.TtoI_down_bias[task_id]
        self.prompt_learner.TtoI.up_proj.weight.data = self.TtoI_up_weight[task_id]
        self.prompt_learner.TtoI.up_proj.bias.data = self.TtoI_up_bias[task_id]
        # for i in range(prompt_depth_text):
        #     self.text_encoder.transformer.resblocks[i].VPT_shallow = torch.nn.Parameter(
        #         self.text_prompt_pool[task_id][i])
        #     self.text_encoder.transformer.resblocks[i].scale_vector_image = torch.nn.Parameter(
        #         self.scale_vector_image[task_id][i])
        #
        # for i in range(prompt_depth_vision):
        #     self.image_encoder.transformer.resblocks[i].VPT_shallow = torch.nn.Parameter(
        #         self.visual_prompt_pool[task_id][i])
        #     self.image_encoder.transformer.resblocks[i].scale_vector = torch.nn.Parameter(
        #         self.scale_vector[task_id][i])
    def contrastive_prompt_loss(image_features, text_features, labels, temp=0.07):
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        logits = image_features @ text_features.t() / temp

        pos_mask = labels.unsqueeze(1) == torch.arange(text_features.size(0), device=labels.device).unsqueeze(0)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss
# --------------------------------------------------------------------
# CIPT
# class ContextPromptLearner(nn.Module):
#     def __init__(self, dataset, clip_model):
#         super().__init__()
#         classnames = dataset.classnames
#         n_cls = len(classnames)
#         n_ctx = 2
#         dtype = clip_model.dtype
#         self.dtype = dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         self.device = next(clip_model.parameters()).device
#
#         # Low-rank decomposition parameters
#         bottleneck_dim = 2
#         self.bottleneck_dim = bottleneck_dim
#
#         # # A' and B' matrices (Eq.6 in paper)
#         self.A_prime = nn.Parameter(torch.empty(n_ctx, bottleneck_dim-1, dtype=dtype))
#         self.B_prime = nn.Parameter(torch.empty(bottleneck_dim-1, ctx_dim, dtype=dtype))
#
#         # Scale vector (w in paper)
#         self.scale_vector = nn.Parameter(torch.randn(ctx_dim, dtype=dtype) * 0.02)
#         self.scale_vector_image = nn.Parameter(torch.randn(ctx_dim, dtype=dtype) * 0.02)
#
#         # # Initialize parameters
#         nn.init.normal_(self.A_prime, std=0.02)
#         nn.init.normal_(self.B_prime, std=0.02)
#
#         # Class-specific prompts
#         classnames = [name.replace("_", " ") for name in classnames]
#         prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         clip_model.cuda()
#
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts.to(self.device)).type(dtype)
#
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts.to(self.device)
#         self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         self.ctx_dim = ctx_dim
#
#         self.projection = nn.Sequential(
#             nn.Linear(clip_model.visual.output_dim, 512),
#             nn.GELU(),
#             nn.LayerNorm(512)
#         ).type(self.dtype)
#
#
    # def construct_prompts(self, ctx, prefix, suffix, label=None):
    #     # dim0 is either batch_size (during training) or n_cls (during testing)
    #     # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
    #     # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
    #     # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
    #
    #     if label is not None:
    #         prefix = prefix[label]
    #         suffix = suffix[label]
    #
    #     prompts = torch.cat(
    #         [
    #             prefix,  # (dim0, 1, dim)
    #             ctx,  # (dim0, n_ctx, dim)
    #             suffix,  # (dim0, *, dim)
    #         ],
    #         dim=1,
    #     )
    #
    #     return prompts
    # def forward(self, context_embeddings=None, image_embeddings=None):
    #     base_prompt = torch.matmul(self.A_prime, self.B_prime)
    #     delta = torch.zeros_like(base_prompt)
    #     if context_embeddings is not None:
    #         context_vector = context_embeddings.mean(dim=(0, 1))
    #         delta += (self.scale_vector * context_vector).unsqueeze(0)  # 逐元素乘并广播
    #
    #     if image_embeddings is not None:
    #         image_vector = image_embeddings.mean(dim=0)  # [D]
    #         delta += (self.scale_vector_image * image_vector).unsqueeze(0)
    #
    #     # Always compute soft prompt
    #     soft_prompt = base_prompt + delta  # (n_ctx, 512)
    #     # soft_prompt = delta
    #     ctx = soft_prompt.unsqueeze(0).expand(self.n_cls, -1, -1)
    #
    #     prefix = self.token_prefix
    #     suffix = self.token_suffix
    #
    #     prompts = self.construct_prompts(ctx, prefix, suffix)
    #
    #     return prompts
#
# class CIPTTextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype
#
#
#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         return x
#
# class CIPT(nn.Module):
#     def __init__(self, dataset):
#         super().__init__()
#         clip_model, _ = clip_origin.load('ViT-B/16', jit=False)
#         self.clip_model = clip_model
#         self.device = next(clip_model.parameters()).device
#         self.prompt_learner = ContextPromptLearner(dataset, clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.image_encoder = clip_model.visual
#         self.text_encoder = CIPTTextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype
#         prototype_feature = torch.empty(11, 512, device=device)
#         self.register_buffer("prototype_feature", prototype_feature)
#         # For getting text embeddings to compute context
#         self.text_embedding = clip_model.token_embedding
#         self.image_embedding = self.prompt_learner.projection
#         with torch.no_grad():
#             self.text_embeddings = self.text_embedding(self.tokenized_prompts.to(self.device)).type(self.dtype)
#     def update_prototype_feature(self, task_id=None):
#
#         if task_id is not None:
#             with torch.no_grad():
#                 prompts = self.prompt_learner.tokenized_prompts
#                 prompts = prompts.to("cuda")
#                 text_features = self.clip_model.encode_text(prompts)
#
#             temp_param = nn.Parameter(text_features.mean(dim=0, keepdim=True))
#
#             temp_param = F.normalize(temp_param, dim=1)
#             self.prototype_feature[task_id] = temp_param
#     def forward(self, image, label=None):
#         torch.cuda.empty_cache()
#         # Get image features
#         image_features = self.image_encoder(image.type(self.dtype))
#
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)# [B, C]
#         image_embedding = self.image_embedding(image_features)
#
#         image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)  # 投影后再次归一化
#         text_embedding = self.text_embeddings.to(image.device)
#         # 文本分支
#         text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # 归一化原始token嵌入
#
#         # Get prompts with context
#         prompts = self.prompt_learner(text_embedding,image_embedding)
#
#         # Encode text
#         text_features = self.text_encoder(prompts, self.tokenized_prompts.to(image.device))
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()  # [B, N]
#         if self.prompt_learner.training:
#             return F.cross_entropy(logits, label)
#
#         return logits

