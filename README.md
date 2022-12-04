# Tuning Free OOD Detection

This repository contains the official code for the paper "Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection". (TODO: Add link once on arxiv)

---

## Setup

First, create a virtual environment for the project (we use Conda to create a python 3.9 environment) and install all the requirments using `requirements.txt`.
1. `conda create -n ood_det python==3.9`
2. `conda activate ood_det`
3. `pip install -r requirements.txt`

Now, create two directories:
1. Data directory
2. Model directory - create two subdirectories: `pretrained_models` and `finetuned_models`

Both directories can be located anywhere but should be specified in `config.py` (See more below).

---
## Running Experiments

At the start of each session, run the following: `. bin/start.sh`

### Running OOD Detection on a Model

Directory paths and training arguments can be specified in `config.py`. Some important arguments are:
- `DATA_DIR`: Path to the data directory
- `MODEL_DIR`: Path to the model directory
- `task_name`: Name of the dataset to use as in-distribution (ID) data.
- `ood_datasets`: List of datasets to use as out-of-distribution (OOD) data.
- `model_class`: Type of model to use. Options are `roberta`, `gpt2`and `t5` (base versions of all models used).
- `do_train`: If true, trains the model on the ID data before performing OOD detection.

After specifying the arguments, run the following command:
`python run_ood_detection.py`


### Training a Model through TAPT

Running the command below will extend the pretraining process on a specified dataset.
Once again, the values in `config.json` can be used to specify the dataset and model to be used.
`python tapt_training/pretrain_roberta.py`

After the new model has been saved to the `pretrained_models` directory within the `MODELS_DIR` directory (specified in `config.py`), it can be used for OOD detection by running `run_ood_detection.py`.

---

## Open Points to Address:
1. Add citations once paper is on arxiv, then make repo public
2. How to share datasets for reproducaibility? Files too heavy for github.  
3. Test refactored versions of umap and tapt



---
## Citation

If you find this repo helpful, welcome to cite our work:
```
(TODO: Add citation once on arxiv)
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