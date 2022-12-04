import os
import sys
import math
import torch
import faiss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaModel, GPT2Model, GPT2PreTrainedModel, T5PreTrainedModel, T5EncoderModel

sys.path.append(os.getcwd() + '/..')
from setup import run_configs

if run_configs.machine == 'local':
    from src.utils.mmd import MMD
else:
    from utils.mmd import MMD


def get_sent_embeddings(model,  input_ids, attention_mask):
    if 'roberta' in model.config._name_or_path:
        outputs = model.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (N, L, D)
        if model.config.sentence_embedding == 'average':
            pooled_outputs = torch.mean(hidden_states, dim=1)  # (N, D)
        else:
            pooled_outputs = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS]) (N, D)

        logits, sent_embedding = model.classifier(pooled_outputs)
        if model.config.layer_representation_for_ood != 'penultimate_layer':
            sent_embedding = pooled_outputs

    elif 'gpt' in model.config._name_or_path or 't5' in model.config._name_or_path:
        outputs = model.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # From last layer of model - (N, L, D)

        if model.config.sentence_embedding == 'average':
            weighted_token_states = (hidden_states * attention_mask.unsqueeze(-1))
            pooled_outputs = weighted_token_states.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)  # Average non-zero tokens
        else:
            # pooled_outputs = hidden_states[:, -1, :]  # Use embedding of last token (Use only when left padding)
            batch_size, sequence_length = input_ids.shape[:2]
            sequence_lengths = torch.ne(input_ids, model.config.pad_token_id).sum(-1) - 1
            pooled_outputs = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        logits, sent_embedding = model.classifier(pooled_outputs)
        if model.config.layer_representation_for_ood != 'penultimate_layer':
            sent_embedding = pooled_outputs

    return logits, sent_embedding


def compute_loss(model, labels, logits, pooled):
    loss, cos_loss = None, None

    if labels is not None:

        if model.config.contrastive_loss == 'margin':
            dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
            mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            mask = mask - torch.diag(torch.diag(mask))
            neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
            max_dist = (dist * mask).max()
            cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (
                        neg_mask.sum(-1) + 1e-3)
            cos_loss = cos_loss.mean()
        else:
            norm_pooled = F.normalize(pooled, dim=-1)  # (N, D); Using L2 normalized embeddings
            cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / model.config.tau)  # (N, N)
            mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
            mask = mask - torch.diag(torch.diag(mask))
            cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)  # Make distribution of scores across each row
            cos_loss = -torch.log(cos_loss + 1e-5)  # 1e-5 for numerical stability
            cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
            cos_loss = cos_loss.mean()

        # Compressed version of HF source, with only relevant losses kept
        if model.num_labels == 1:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))

        loss = (model.config.cross_entropy_weight * loss) + (model.config.contrastive_weight * cos_loss)

        if model.config.contrastive_weight == 0.0:
            cos_loss = torch.tensor(0)

    return loss, cos_loss


def get_bank(dataloader, model):
    bank = None         # Concatenation of all pooled outputs (i.e. penultimate layer representations)
    label_bank = None   # Concatenations of labels

    for batch in dataloader:
        model.eval()
        batch = {key: value.to(run_configs.device) for key, value in batch.items()}
        labels = batch['labels']
        logits, pooled = get_sent_embeddings(model, batch['input_ids'], batch['attention_mask'])

        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach()
        else:
            bank = torch.cat([pooled.clone().detach(), bank], dim=0)
            label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0)

    return bank, label_bank


def prepare_ood(model, is_train, dataloader=None):

    if is_train:
        model.bank, model.label_bank = get_bank(dataloader, model)

        model.norm_bank = F.normalize(model.bank, dim=-1)         # Normalized penultimate layer pooled outputs
        N, d = model.bank.size()
        model.all_classes = list(set(model.label_bank.tolist()))  # List of class labels

        model.class_mean = torch.zeros(max(model.all_classes) + 1, d).to(run_configs.device)
        for c in model.all_classes:
            model.class_mean[c] = (model.bank[model.label_bank == c].mean(0))

        centered_bank = (model.bank - model.class_mean[model.label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(
            np.float32)  # .precision_ is the estimated pseudo-inverse matrix.
        model.class_var = torch.from_numpy(precision).float().to(run_configs.device)

        # Inter-class Dispersion and Intra-class Compactness from CIDER
        prototype_embeddings = []
        for c in model.all_classes:
            mu_c = torch.mean(model.bank[model.label_bank == c], dim=0)  # Average of all embeddings for class c
            mu_c = F.normalize(mu_c, dim=-1)  # L2 norm
            prototype_embeddings.append(mu_c)
        prototype_embeddings = torch.stack(prototype_embeddings)         # Convert to (C, D) tensor

        class_cosine_sim = prototype_embeddings @ prototype_embeddings.T  # (C,C) each element is the cosine sim between 2 class prototypes
        class_cosine_sim = np.degrees(np.arccos(
            class_cosine_sim.detach().cpu().numpy()))  # Inverse cos - converts cosine sim to radians and then degrees
        class_cosine_sim = torch.triu(
            torch.tensor(class_cosine_sim))  # Keeps the upper triangle matrix, since the matrix is symmetric
        class_cosine_sim.fill_diagonal_(0)   # Set diagonal elements to 0, since those are cosine sim of class with itself

        num_classes = class_cosine_sim.shape[0]
        dispersion = (2 / (num_classes * (num_classes - 1))) * torch.sum(class_cosine_sim)
        model.dispersion = dispersion.detach().cpu().numpy()

        # Compactness - cosine sim of each z_i from class j, with mu_j (i.e. prototype of class j)
        model.compactness = 0
        if not run_configs.debug_mode:
            for j in model.all_classes:
                z_i = model.norm_bank[model.label_bank == j]    # (n, D)
                mu_j = prototype_embeddings[j]                  # (D,)
                cos_sim = z_i @ mu_j
                degree_sim = np.degrees(
                    np.arccos(cos_sim.detach().cpu().numpy()))  # Convert cosine similarity to degrees
                model.compactness += torch.mean(torch.tensor(degree_sim))
            model.compactness = (model.compactness / num_classes).detach().cpu().numpy()

        # kNN
        model.index = faiss.index_factory(model.config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        z = model.bank.detach().clone().cpu().numpy()
        faiss.normalize_L2(z)
        model.index.add(z)
    else:

        bank_val, label_bank_val = get_bank(dataloader, model)

        norm_bank_val = F.normalize(bank_val, dim=-1)  # Normalized penultimate layer pooled outputs
        N, d = bank_val.size()
        all_classes_val = list(set(label_bank_val.tolist()))  # List of class labels

        class_mean_val = torch.zeros(max(all_classes_val) + 1, d).to(run_configs.device)
        for c in all_classes_val:
            class_mean_val[c] = (bank_val[label_bank_val == c].mean(0))

        # Average similarity of ID points from closest class centroids (for ID-OOD seperability)
        normalized_class_mean = F.normalize(class_mean_val, dim=-1)
        id_cosine_sim = norm_bank_val @ normalized_class_mean.T   # Cosine sim of each point with each class centroid
        id_cosine_sim = id_cosine_sim.max(-1).values              # Cosine sim of each point with closest class centroid
        id_cosine_sim = np.degrees(np.arccos(id_cosine_sim.detach().cpu().numpy()))  # Convert to degrees
        model.id_cosine_sim = torch.tensor(id_cosine_sim).mean()  # Average across all points

        model.norm_bank_val = norm_bank_val


def compute_ood(model, input_ids=None, attention_mask=None, labels=None):
    logits, pooled = get_sent_embeddings(model, input_ids, attention_mask)

    if model.pooled_ood is None:
        model.pooled_ood = F.normalize(pooled.clone().detach(), dim=-1)
    else:
        model.pooled_ood = torch.cat([F.normalize(pooled.clone().detach(), dim=-1), model.pooled_ood], dim=0)

    # Average similarity of OOD points from closest class centroids (for ID-OOD seperability)
    normalized_class_mean = F.normalize(model.class_mean, dim=-1)
    ood_cosine_sim = F.normalize(pooled.clone().detach(), dim=-1) @ normalized_class_mean.T  # Cosine sim of each point with each class centroid
    ood_cosine_sim = ood_cosine_sim.max(-1).values                                           # Cosine sim of each point with closest class centroid
    ood_cosine_sim = np.degrees(np.arccos(ood_cosine_sim.detach().cpu().numpy()))            # Convert to degrees
    if model.total_ood_cosine_sim is None:
        model.total_ood_cosine_sim = torch.tensor(ood_cosine_sim)                            # Since we are working with batches, only sum for now
    else:
        model.total_ood_cosine_sim = torch.cat([torch.tensor(ood_cosine_sim), model.total_ood_cosine_sim], dim=0)

    # Softmax/ MSP Score
    m = torch.nn.Softmax(dim=-1).cuda()
    softmax_score, _ = torch.max(m(logits), dim=-1)

    # Energy Score
    temperature = 1
    energy_score = temperature * torch.logsumexp(logits / temperature, dim=1)

    # Mahalanobis Distance Score
    maha_score = []
    for c in model.all_classes:
        centered_pooled = pooled - model.class_mean[c].unsqueeze(0)
        ms = torch.diag(centered_pooled @ model.class_var @ centered_pooled.t())  # @ is traditional matrix multiplication, also np.matmul()
        maha_score.append(ms)
    maha_score = torch.stack(maha_score, dim=-1)
    maha_score = maha_score.min(-1)[0]
    maha_score = -maha_score

    # kNN from my code
    k = 1
    z = pooled.detach().clone().cpu().numpy()
    faiss.normalize_L2(z)
    scores, _ = model.index.search(z, 10000)
    scores[scores < 1e-20] = 0   # To avoid underflow for k-avg NN
    knn_distances = -1 * (1 - scores[:, k-1])

    ood_keys = {}
    ood_keys['softmax'] = softmax_score.tolist()
    ood_keys['energy'] = energy_score.tolist()
    ood_keys['maha'] = maha_score.tolist()
    ood_keys['kNN'] = knn_distances.tolist()  # Consistent similarity scores with cosine

    if model.config.report_all_metrics:

        # Cosine Similarity Score; Alternate way of calculating kNN with cosine distance
        k = 1
        norm_pooled = F.normalize(pooled, dim=-1)  # Even kNN does normalization if layernorm not already present
        cosine_similarity = norm_pooled @ model.norm_bank.t()
        cosine_similarity = cosine_similarity.cpu()
        cosine_similarity = torch.tensor(np.sort(cosine_similarity, axis=1))  # Sorted in ascending order per row
        nearest_nn = cosine_similarity[:, -k]
        ood_keys['cosine'] = nearest_nn.tolist()

        for k in [10, 50, 100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000]:
            ood_keys[f'{k}-NN-cosine'] = cosine_similarity[:, -k].tolist()
            ood_keys[f'{k}-avg-NN-cosine'] = np.average(cosine_similarity[:, -k:], axis=1).tolist()

            ood_keys[f'{k}-NN'] = (-1 * (1 - scores[:, k-1])).tolist()
            ood_keys[f'{k}-avg-NN'] = (-1 * (1 - np.average(scores[:, :k], axis=1))).tolist()

    return ood_keys


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # From HF source code, included here for reference
        # https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/roberta/modeling_roberta.py#L1435
        # classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(classifier_dropout)

        '''
        From auto HF architecture:
        (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True))
        '''

    def forward(self, features):
        # x = features[:, 0, :] # take <s> token (equiv. to [CLS])
        x = self.dropout(features) # features is now (N, L, D) not (N, D)
        x = self.dense(x)
        x = pooled = torch.tanh(x) # Only addition from HF code is variable pooled
        x = self.dropout(x)
        x = self.out_proj(x)

        return x, pooled


class RobertaForSequenceClassification(BertPreTrainedModel):
    # HF Source: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/roberta/modeling_roberta.py#L1162
    #            https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/roberta/modeling_roberta.py#L693

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = ClassificationHead(config)

        # HF code instead does self.post_init() which is self.init_weights() + self._backward_compatibility_gradient_checkpointing()
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        logits, pooled = get_sent_embeddings(self, input_ids, attention_mask)
        loss, cos_loss = compute_loss(self, labels, logits, pooled)

        output = (logits,) + outputs[2:]    # At least during inference, outputs[2:] is ()
        output = output + (pooled,)         # Concat tuple, but why?
        return ((loss, cos_loss) + output) if loss is not None else output
        # # From HF source code
        # output = (logits,) + outputs[2:]
        # return ((loss,) + output) if loss is not None else output


class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    # HF Source: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/gpt2/modeling_gpt2.py#L1330
    #            https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/gpt2/modeling_gpt2.py#L669

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        # self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)  # This is classifier head
        self.classifier = ClassificationHead(config)

        # HF code instead does self.post_init() which is self.init_weights() + self._backward_compatibility_gradient_checkpointing()
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        logits, pooled = get_sent_embeddings(self, input_ids, attention_mask)
        loss, cos_loss = compute_loss(self, labels, logits, pooled)

        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        return ((loss, cos_loss) + output) if loss is not None else output


class T5ForSequenceClassification(T5PreTrainedModel):
    # HF Source: https://github.com/huggingface/transformers/blob/v4.22.2/src/transformers/models/t5/modeling_t5.py#L1760

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.transformer = T5EncoderModel(config)
        self.classifier = ClassificationHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        logits, pooled = get_sent_embeddings(self, input_ids, attention_mask)
        loss, cos_loss = compute_loss(self, labels, logits, pooled)

        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        return ((loss, cos_loss) + output) if loss is not None else output
