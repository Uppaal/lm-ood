import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass, field
from setup import run_configs, paths, logger

sys.path.extend([(paths.root_path + source_path) for source_path in os.listdir(paths.root_path) if not (paths.root_path + source_path) in sys.path])
debug_mode = run_configs.debug_mode

from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    set_seed, EvalPrediction, Trainer, DataCollatorForTokenClassification, AutoConfig,
    AutoModel, AutoTokenizer,
    TrainingArguments, default_data_collator, DataCollatorWithPadding, MODEL_WITH_LM_HEAD_MAPPING)
from datasets import load_metric, load_dataset
import datasets



@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    use_auth_token: bool = field(default=False, metadata={"help": ""})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": ""})

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The input training data files (multiple files in glob format). "
                "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
            )
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    # train_ref_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    # )
    # eval_ref_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    # )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Ratio of length of a span of masked tokens to surrounding context length for permutation language"
                " modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    max_seq_length: int = field(default=512, metadata={"help": "Maximum length of sequence"})
    data_cache_dir: str = field(default=None)
    pad_to_max_length: bool = field(default=True)
    model_max_length: int = field(default=512, metadata={"help": "Should be tokenizer.model_max_length"})
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List, config):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task, config)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task, config):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "token_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels)
            # return _TokenClassificationHead(encoder_hidden_size, task.num_labels, config)
        elif task.type == "masked_language_modelling":
            return MLMHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_ids=None,
            **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs

#TODO MLM head
class MLMHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss
class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_ = self.classifier(sequence_output)
        sequence_output_dropout = self.dropout(sequence_output_)
        logits = self.out_proj(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
class _TokenClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        # super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.config = config

        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, encoder_outputs, labels=None, return_dict=True, **kwargs):

        sequence_output = encoder_outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if preds.ndim == 2:
        # Token classification
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    elif preds.ndim == 3:
        # Sequence classification
        metric = load_metric("seqeval")

        predictions = np.argmax(preds, axis=2)

        true_predictions = [
            [f"tag-idx-{p}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]
        true_labels = [
            [f"tag-idx-{l}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]

        # Remove ignored index (special tokens)
        results = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        raise NotImplementedError()


def main(model_args, data_args, training_args):

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True if 'roberta' in model_args.encoder_name_or_path else False
    )

    tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(paths.MODEL_DIR + '/pretrained_models/roberta_base', output_hidden_states=True) # num_labels= TODO add
    model = MultiTaskModel(model_args.encoder_name_or_path, tasks, config=config)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.samples_to_keep is not None:
            train_dataset = train_dataset.select(range(data_args.samples_to_keep))

    if training_args.do_eval:
        if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(data_args.max_eval_samples)))

            eval_datasets = new_ds

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.samples_to_keep
            if data_args.samples_to_keep is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "token_classification":
                data_collator = DataCollatorForTokenClassification(
                    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
                )
            else:
                if data_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                        tokenizer, pad_to_multiple_of=8
                    )
                else:
                    data_collator = None

            trainer.data_collator = data_collator
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_datasets)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)



def tokenize_seq_classification_dataset(tokenizer, raw_datasets, task_id, data_args, training_args):
    sentence1_key, sentence2_key = "sentence1", "sentence2"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_text(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        examples["labels"] = examples.pop("label")
        result["task_ids"] = [task_id] * len(examples["labels"])
        return result

    def tokenize_and_pad_text(examples):
        result = tokenize_text(examples)

        examples["labels"] = [
            [l] + [-100] * (max_seq_length - 1) for l in examples["labels"]
        ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["idx", sentence1_key, sentence2_key]
        train_dataset = raw_datasets["train"].map(
            tokenize_and_pad_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )
        validation_dataset = raw_datasets["validation"].map(
            tokenize_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )

    return train_dataset, validation_dataset

def load_seq_classification_dataset(task_id, tokenizer, data_args, training_args):

    task_name = "mrpc"
    raw_datasets = load_dataset("glue", task_name, cache_dir=data_args.data_cache_dir)

    num_labels = len(raw_datasets["train"].features["label"].names)

    train_dataset, validation_dataset = tokenize_seq_classification_dataset(
        tokenizer,
        raw_datasets,
        task_id,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id, name=task_name, num_labels=num_labels, type="seq_classification"
    )

    return train_dataset, validation_dataset, task_info

def tokenize_token_classification_dataset(raw_datasets, tokenizer, task_id, label_list, text_column_name, label_column_name, data_args, training_args,):

    label_to_id = {i: i for i in range(len(label_list))}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # if data_args.label_all_tokens:
                    #     label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    # else:
                    #     label_ids.append(-100)
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens"]

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=1,  #data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
        )

    return tokenized_datasets

def load_token_classification_dataset(task_id, tokenizer, data_args, training_args):

    dataset_name = "conll2003"
    raw_datasets = load_dataset(dataset_name)

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = raw_datasets["train"].features[label_column_name].feature.names
    num_labels = len(label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name=dataset_name,
        num_labels=num_labels,
        type="token_classification",
    )

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )

def load_datasets(tokenizer, data_args, training_args):
    (
        seq_classification_train_dataset,
        seq_classification_validation_dataset,
        seq_classification_task,
    ) = load_seq_classification_dataset(0, tokenizer, data_args, training_args)
    (
        token_classification_train_dataset,
        token_classification_validation_dataset,
        token_classification_task,
    ) = load_token_classification_dataset(1, tokenizer, data_args, training_args)

    # Merge train datasets
    train_dataset_df = seq_classification_train_dataset.to_pandas().append(
        token_classification_train_dataset.to_pandas()
    )
    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    train_dataset.shuffle(seed=123)

    # Append validation datasets
    validation_dataset = [
        seq_classification_validation_dataset,
        token_classification_validation_dataset,
    ]

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    tasks = [seq_classification_task, token_classification_task]
    return tasks, dataset



if __name__ == "__main__":
    model_args = ModelArguments(encoder_name_or_path="roberta-base")
    data_args = DataTrainingArguments(max_seq_length=128, max_train_samples=10, max_eval_samples=10)
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=f"{paths.temp_output_dir}/test",
        learning_rate=2e-5,
        num_train_epochs=1,
        overwrite_output_dir=True,
    )
    main(model_args, data_args, training_args)

