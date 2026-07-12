import yaml
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel, RobertaForMaskedLM, T5ForConditionalGeneration

repo_root = Path(__file__).resolve().parent
with open(repo_root / "configs.yaml", "r") as f:
    config = yaml.safe_load(f)
if config is None:
    raise ValueError("configs.yaml is empty")

def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path

resolve_path(config["paths"]["DATA_DIR"]).mkdir(parents=True, exist_ok=True)

model_dir = resolve_path(config["paths"]["MODEL_DIR"])
pretrained_dir = model_dir / "pretrained_models"
pretrained_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "finetuned_models").mkdir(parents=True, exist_ok=True)

models = [
    ("roberta-base", "roberta_base", RobertaForMaskedLM),
    ("gpt2", "gpt2_base", GPT2LMHeadModel),
    ("t5-base", "t5_base", T5ForConditionalGeneration),
]

for hub_id, local_name, model_cls in models:
    output_dir = pretrained_dir / local_name
    print(f"Downloading {hub_id} -> {output_dir}")
    AutoTokenizer.from_pretrained(hub_id, use_fast=True).save_pretrained(output_dir)
    model_cls.from_pretrained(hub_id).save_pretrained(output_dir)
