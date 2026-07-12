from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig


def load_model(model_path, num_labels=None, load_only_body=False):

        # model_path = exp_configs.MODEL_DIR + f'/{model_name}'
        if num_labels is not None:
            model_config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, output_hidden_states=True)
        else:
            model_config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        if load_only_body:
            model = AutoModelForMaskedLM.from_pretrained(model_path, config=model_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)
        return model


def load_tokenizer(model_path, model_str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if model_str == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer
