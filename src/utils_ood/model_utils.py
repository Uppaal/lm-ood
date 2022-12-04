import os
import sys
from setup import run_configs, paths
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
sys.path.extend([(paths.root_path + source_path) for source_path in os.listdir(paths.root_path) if not (paths.root_path + source_path) in sys.path])

# GPT2 Model from: https://huggingface.co/gpt2
# DistilRoBERTa-base from: https://huggingface.co/distilroberta-base

def load_model(model_name, num_labels=None, load_only_body=False):
        # Note: "max_position_embeddings": 514, unlike BERT (512); but embedding size is still 512: https://github.com/pytorch/fairseq/issues/1187

        model_path = paths.MODEL_DIR + f'/{model_name}'
        if num_labels is not None:
            model_config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, output_hidden_states=True)
        else:
            model_config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        if load_only_body:
            # Ref: https://github.com/huggingface/transformers/issues/8901
            model = AutoModelForMaskedLM.from_pretrained(model_path, config=model_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)
        # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # return model, tokenizer
        return model


def load_tokenizer(model_str):
    if model_str == 'roberta':
        model_name = '/pretrained_models/roberta_base'
    elif model_str == 'gpt2':
        model_name = '/pretrained_models/gpt2_base'  # gpt2-small; renamed to base for convenience
    elif model_str == 't5':
        model_name = '/pretrained_models/t5_base'

    model_path = paths.MODEL_DIR + f'/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if model_str == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # "left" is common, but not advised
    return tokenizer

# https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd


'''
(output): RobertaOutput(
    (dense): Linear(in_features=3072, out_features=768, bias=True)
(LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
(dropout): Dropout(p=0.1, inplace=False)

(lm_head): RobertaLMHead(
(dense): Linear(in_features=768, out_features=768, bias=True)
(layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
(decoder): Linear(in_features=768, out_features=50265, bias=True)
)

(classifier): RobertaClassificationHead(
(dense): Linear(in_features=768, out_features=768, bias=True)
(dropout): Dropout(p=0.1, inplace=False)
(out_proj): Linear(in_features=768, out_features=2, bias=True)
)
'''


load_model()

