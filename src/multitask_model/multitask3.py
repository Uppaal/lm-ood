# import json
# import pandas as pd
import numpy as np
# import transformers
# from datasets import load_dataset, dataset_dict
# import nlp
# import dataclasses
import logging
import transformers
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import InputDataClass, DefaultDataCollator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict

from utils.dataset_utils import DatasetUtil

logging.basicConfig(level=logging.INFO)


class MultitaskModel(transformers.AutoModel):#PreTrainedModel): # TODO

    def __init__(self, encoder, taskmodels_dict):

        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        # super().__init__(transformers.AutoConfig())#PretrainedConfig()) #TODO
        super().__init__(transformers.AutoConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create_model(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        task_models_dict = {}
        for task, model_type in model_type_dict.items():
            print(task)
            print(model_type)
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            task_models_dict[task] = model
        return cls(encoder=shared_encoder, taskmodels_dict=task_models_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):

        """
        Each encoder has its attributes according to model architecture: BERT, Roberta,Alberta
        This function gets attribute of the encoder.
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith('Bert'):
            return 'bert'
        if model_class_name.startswith('Roberta'):
            return 'roberta'
        if model_class_name.startswith('Albert'):
            return 'albert'
        if model_class_name.startswith():
            return

        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task, **kwargs):
        return self.taskmodels_dict[task](**kwargs)

# Class to load data with its task name. Decorator for changing Dataloader function to use a task name
class DataLoaderTaskname:
    def __init__(self, task, data_loader):
        self.task = task
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task"] = StrIgnoreDevice(self.task)
            yield batch

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self

# Class to combine several data loaders into a single "data loader"
class MultitaskDataLoader:

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {task: len(dataloader) for task, dataloader in self.dataloader_dict.items()}
        self.task_lst = list(self.dataloader_dict)
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloader_dict.values())

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and get a batch from the respective task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task in enumerate(self.task_lst):
            task_choice_list += [i] * self.num_batches_dict[task]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task: iter(dataloader)
            for task, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task = self.task_lst[task_choice]
            yield next(dataloader_iter_dict[task])

# class to set up the trainer
class MultitaskTrainer(transformers.Trainer):

    def single_task_dataloader(self, task, train_dataset):
        """
        returns the single task data loader of a given task
        """
        if self.train_dataset is None:
            raise ValueError("Trainer needs a dataset...:(")
        train_sampler = (RandomSampler(train_dataset) if self.args.local_rank == -1
                         else DistributedSampler(train_dataset))

        data_loader = DataLoaderTaskname(task=task, data_loader=DataLoader(train_dataset,
                                                                           batch_size=self.args.train_batch_size,
                                                                           sampler=train_sampler,
                                                                           collate_fn=self.data_collator.collate_batch
                                                                           ))
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataLoader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataLoader({
            task: self.single_task_dataloader(task, task_dataset)
            for task, task_dataset in self.train_dataset.items()})



# Class to sample bacth from the featurized datasets, The Datac is extended to work with batches
# Data collators are objects that will form a batch by using a list of dataset elements as input
class NLPDataCollator(DefaultDataCollator):
    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        # features - list of batch_size len. Each item is a dict of input_ids, attention_mask and labels tensors.
        first = features[0] # First dp
        batch = {}
        # if isinstance(first, dict):

        # featurized dataset are in the form of list of dictionaries
        # Adapt the DataCollator to have a list of dictionary
        if "labels" in first and first["labels"] is not None:
            # batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
            batch["labels"] = torch.stack([f['labels'] for f in features])
        for k, v in first.items():
            if k != "labels" and v is not None and not isinstance(v, str):
                batch[k] = torch.stack([f[k] for f in features])
        return batch
        # else:
        #     print("OHNOOHNOOHNOOHNOOHNOOHNOOHNO")
        #     # otherwise, revert to using the default collate_batch
        #     return DefaultDataCollator().collate_batch(features)


def save_model(model_name, multitask_model):
    print(
        f"In save model model_name {model_name}"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["spaadia_squad_pairs", "quora_keyword_pairs"]:
        multitask_model.taskmodels_dict[task_name].config.to_json_file(
            f"../temp_outputs/{task_name}_model/config.json"
        )
        torch.save(
            multitask_model.taskmodels_dict[task_name].state_dict(),
            f"../temp_outputs/{task_name}_model/pytorch_model.bin",
        )
        tokenizer.save_pretrained(f"../temp_outputs/{task_name}_model/")



max_length = 256
max_train_samples = 5

dataset_name = '20newsgroups'
data_util = DatasetUtil(dataset_name=dataset_name)
dataset = data_util.get_dataset(dataset_name, split=None)
num_id_labels = len(data_util.id_classes)

model_name = 'roberta-base'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
multitask_model = MultitaskModel.create_model(model_name=model_name,
                                              model_type_dict={
                                                  "20NG-classification": transformers.AutoModelForSequenceClassification,
                                                  "20NG-mlm": transformers.AutoModelForMaskedLM
                                              },
                                              model_config_dict={
                                                  "20NG-classification": transformers.AutoConfig.from_pretrained(model_name,num_labels=num_id_labels),
                                                  "20NG-mlm": transformers.AutoConfig.from_pretrained(model_name)
                                              })

task_list = ["20NG-classification", "20NG-mlm"]
ft_train_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][0]].select(range(max_train_samples)))
ft_val_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][1]].select(range(max_train_samples)))
mlm_train_dataset = data_util.get_tensors_for_mlm(dataset[data_util.dataset_to_split[dataset_name][0]].select(range(max_train_samples)))
mlm_val_dataset = data_util.get_tensors_for_mlm(dataset[data_util.dataset_to_split[dataset_name][1]].select(range(max_train_samples)))

#Loading datasets
# dataset_dict = {
#     "mnli": datasets.load_dataset('glue', 'mnli'),
#     "stsb_multi_mt": datasets.load_dataset("stsb_multi_mt",name="en")
# }

# def convert_to_mnli_inputs(example_batch):
#     inputs = list(zip(example_batch['premise'], example_batch['hypothesis']))
#     features = tokenizer.batch_encode_plus(inputs, max_length=max_length,
#                                            pad_to_max_length=True, truncation=True)
#     features['labels'] = example_batch['label']
#     return features
#
# def convert_to_stsb_multi_mt_inputs(example_batch):
#     inputs = example_batch['sentence1']
#     features = tokenizer.batch_encode_plus(inputs, max_length=max_length,
#                                            pad_to_max_length=True, truncation=True)
#     # features['labels'] = example_batch['sentence1']
#     return features
#     # inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
#     # features = tokenizer.batch_encode_plus(inputs, max_length=max_length,
#     #                                        pad_to_max_length=True, truncation=True)
#     # features['labels'] = example_batch['similarity_score']
#     # return features

# Construct the featurized input data
# featurized_funct_dict = {
#     "mnli": convert_to_mnli_inputs,
#     "stsb_multi_mt": convert_to_stsb_multi_mt_inputs
# }
#
# column_dict = {
#     "20NG-classification": ['input_ids', 'attention_mask', 'labels'],
#     "20NG-mlm": ['input_ids', 'attention_mask']
# }
#
# Featurizing datasets
# features_dict = {}
# for task, dataset in dataset_dict.items():
#     print("--------------task---------:", task)
#     features_dict[task] = {}
#     for phase, phase_dataset in dataset.items(): # phase and phase_dataset are specific splits
#         phase_dataset = phase_dataset.select(range(max_train_samples))
#         features_dict[task][phase] = phase_dataset.map(featurized_funct_dict[task],
#                                                        batched=True,
#                                                        load_from_cache_file=False)
#         print(task, phase, len(phase_dataset), len(features_dict[task][phase]))
#         features_dict[task][phase].set_format(
#             type='torch',
#             columns=column_dict[task]
#         )
#         logger.info(f'Task: {task}, Split:{phase}, Len:{len(phase_dataset)}, Len:{len(features_dict[task][phase])}')
#     # features_dict[task][split] is tokenized torch tensors (don't make dataloader though)

features_dict = {task: {} for task in task_list}
features_dict[task_list[0]]['train'] = ft_train_dataset
features_dict[task_list[0]]['val'] = ft_val_dataset
features_dict[task_list[1]]['train'] = mlm_train_dataset
features_dict[task_list[1]]['val'] = mlm_val_dataset

train_dataset = {task: dataset["train"] for task, dataset in features_dict.items()}

trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir="../../temp_outputs/multitask_model",
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=1,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=8,
        save_steps=3000,
    ),
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset,
)
trainer.train()

preds_dict = {}
for task in ["mnli", "commonsense_qa", "stsb_multi_mt"]:
    eval_dataloader = DataLoaderTaskname(
        task,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task]["validation"])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task] = trainer._prediction_loop(
        eval_dataloader,
        description=f"Validation: {task}",
    )




