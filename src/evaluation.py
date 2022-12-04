import torch
import numpy as np
import sklearn.metrics as sk
from sklearn.metrics import roc_auc_score

from model import compute_ood
from src.utils.mmd import MMD


def merge_keys(l, keys):
    new_dict = {}

    for key in keys:
        new_dict[key] = []
        for batch_scores in l:
            new_dict[key] += batch_scores[key]
    return new_dict


def evaluate_ood(args, model, id_dataloader, ood_dataloader, tag):
    keys = ['softmax', 'maha', 'cosine', 'energy', 'kNN', 'k-avg-NN']

    in_scores = []
    model.pooled_ood = None
    model.total_ood_cosine_sim = None

    for batch in id_dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = compute_ood(model=model, input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'])
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, list(in_scores[0].keys()))

    out_scores = []
    model.pooled_ood = None
    model.total_ood_cosine_sim = None
    for batch in ood_dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = compute_ood(model=model, input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'])
            out_scores.append(ood_keys)
    out_scores = merge_keys(out_scores, list(out_scores[0].keys()))

    outputs = {}

    mmd = MMD(model.norm_bank_val, model.pooled_ood, kernel='linear')
    outputs[f'{tag}_mmd'] = mmd

    id_ood_seperability = (model.total_ood_cosine_sim.mean() - model.id_cosine_sim).detach().cpu().numpy()
    outputs[f'{tag}_id-ood-seperability'] = id_ood_seperability

    for key in list(ood_keys.keys()):
        ins = np.array(in_scores[key], dtype=np.float64)
        outs = np.array(out_scores[key], dtype=np.float64)
        inl = np.ones_like(ins).astype(np.int64)
        outl = np.zeros_like(outs).astype(np.int64)
        scores = np.concatenate([ins, outs], axis=0)
        labels = np.concatenate([inl, outl], axis=0)
        labels_rev = 1-labels # Flip 0s and 1s

        aupr_in = sk.average_precision_score(labels, scores)
        aupr_out = sk.average_precision_score(labels_rev, -scores)
        auroc = sk.roc_auc_score(labels, scores)
        fpr = fpr_and_fdr_at_recall(labels, scores, recall_level=0.95)

        outputs[f'{tag}_{key}'] = {'AUROC': auroc, 'AUPR-IN': aupr_in, 'AUPR-OUT': aupr_out, 'FPR95':fpr}

    return outputs


def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    score = fpr_and_fdr_at_recall(new_key, prediction)
    return score


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1] # Translates slice objects to concatenation along the first axis.

    tps = stable_cumsum(y_true)[threshold_idxs] # Cumulative sum
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))
