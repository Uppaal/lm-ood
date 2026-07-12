# Tuning Free OOD Detection

This repository contains the official code for the paper [Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection](https://aclanthology.org/2023.acl-long.717/) (ACL 2023).

_Update July 2026:_ This codebase was recently updated to make it easier to install and run. Those updates may cause small differences from the exact numbers reported in the paper, but the trends and findings remain the same, and observed numerical changes have been minor.

## Setup

Create a Python 3.9 environment, and install the required packages by running `pip install -r requirements.txt`.

Next, fill the required values in `configs.yaml`. Some important arguments are:
- `DATA_DIR`: Path to the data directory
- `MODEL_DIR`: Path to the model directory; all trained checkpoints are saved here.
- `task_name`: Name of the dataset to use as in-distribution (ID) data.
- `ood_datasets`: List of datasets to use as out-of-distribution (OOD) data.
- `model_class`: Type of model to use. Options are `roberta`, `gpt2`, and `t5`.
- `model_path`: Required local checkpoint path.
- `do_train`: If true, trains the model on the ID data before performing OOD detection.

With `do_train: false`, OOD detection runs directly on `model_path`. With `do_train: true`, training starts from `model_path`, saves to `MODEL_DIR/finetuned_models/{model_class}_base_{task_name}`, and OOD detection runs on the in-memory fine-tuned model.

When `do_train: true`, `run_ood_detection.py` uses a weighted sum of cross-entropy and contrastive loss:
```yaml
# CE-only fine-tuning
cross_entropy_weight: 1.0
contrastive_weight: 0.0

# CE + SupCon fine-tuning
cross_entropy_weight: 1.0
contrastive_weight: 2.0
contrastive_loss: supcon

# SupCon-only fine-tuning
cross_entropy_weight: 0.0
contrastive_weight: 1.0
contrastive_loss: supcon
```
`tau` controls the SupCon temperature. These loss settings apply to `run_ood_detection.py`, not the TAPT scripts.


Finally, run the following script to create your data and model directories, and download pretrained models for evaluation (`roberta_base`, `gpt2_base`, and `t5_base`). 
```bash
python download_models.py
```


## Running OOD Detection on a Model

```bash
python run_ood_detection.py
```


## Training a Model through TAPT

TAPT is a two-step process: first run masked language modeling on the ID data, then run supervised cross-entropy fine-tuning from that TAPT checkpoint.

The first stage of training stores a checkpoint in `MODEL_DIR/pretrained_models/{model_class}_base_{task_name}_tapt`.
```bash
python tapt_training/pretrain_roberta.py
```

The second stage loads that TAPT checkpoint and saves the final classifier to `MODEL_DIR/finetuned_models/{model_class}_base_{task_name}`.
```
python tapt_training/finetune_roberta.py
```

For evaluation, set `model_path` in `config.yaml` to the final classifier checkpoint before running `run_ood_detection.py`.

---

## Citation

If you find this repo helpful, you are welcome to cite our work:
```
@inproceedings{uppaal-etal-2023-fine,
    title = "Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection",
    author = "Uppaal, Rheeya  and Hu, Junjie  and Li, Yixuan",
    editor = "Rogers, Anna  and Boyd-Graber, Jordan  and Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.717",
    doi = "10.18653/v1/2023.acl-long.717",
    pages = "12813--12832",
}
```
Our codebase borrows from the following:

```
@inproceedings{zhou2021contrastive,
  title={Contrastive Out-of-Distribution Detection for Pretrained Transformers},
  author={Zhou, Wenxuan and Liu, Fangyu and Chen, Muhao},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={1100--1111},
  year={2021}
}

@article{liu2020tfew,
  title={Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning},
  author={Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin},
  journal={arXiv preprint arXiv:2205.05638},
  year={2022}
}
```
