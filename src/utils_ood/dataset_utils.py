import os
import sys
import json
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset

from vizualization.plot_utils import plot_bar
from setup import run_configs, paths, logger

sys.path.extend([(paths.root_path + source_path) for source_path in os.listdir(paths.root_path) if not (paths.root_path + source_path) in sys.path])


class DatasetUtil():
    def __init__(self, dataset_name, hf_cache_dir=None, max_length=100, tokenizer=None):
        '''
        Sets paths, params and metadata for all datasets.
        :param dataset_name:
        :param lm_model:
        :param model_dir:
        :param hf_cache_dir:
        :param max_length:
        :param debug_mode:
        :tokenizer_str: str including model tokenizer name. 'roberta', 'gpt'
        '''
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.hf_cache_dir = hf_cache_dir

        if self.hf_cache_dir is None:
            self.hf_cache_dir = f'{paths.DATA_DIR}/huggingface_datasets' if run_configs.machine == 'galaxy' else ''

        self.tokenizer = tokenizer
        # if tokenizer_str is None:
        #     logger.info(f'No tokenizer specified. Loading default RoBERTa tokenizer.')
        #     tokenizer_str = 'roberta'
        #
        # if tokenizer_str == 'roberta':
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         paths.MODEL_DIR_local + '/pretrained_models/roberta_base', use_fast=True)
        # elif tokenizer_str == 'gpt2':
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         paths.MODEL_DIR_local + '/pretrained_models/gpt2_base', use_fast=True)
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        self.add_dataset_metadata()


    def get_dataset(self, dataset_name, split):
        # Load dataset: dataset['train'] and dataset['test'] are iterable list of dicts: {'text', 'label'}
        logger.info(f'Loading {split} split of {dataset_name} dataset...')

        if dataset_name in ['imdb', 'yelp_polarity', 'snli', 'dbpedia_14']:
            return load_dataset(dataset_name, split=split, cache_dir=self.hf_cache_dir)

        if dataset_name in ['sst2', 'rte', 'stsb']:
            return load_dataset('glue', dataset_name, split=split, cache_dir=self.hf_cache_dir)

        if dataset_name in ['record']:
            return load_dataset('super_glue', dataset_name, split=split, cache_dir=self.hf_cache_dir)

        if dataset_name == 'mnli':
            dataset = load_dataset('glue', dataset_name, split=split, cache_dir=self.hf_cache_dir)
            processed_dataset = DatasetDict()
            processed_dataset['train'] = dataset['train']
            processed_dataset['val'] = self.merge_dataset_splits([dataset['validation_matched'],
                                                                  dataset['validation_mismatched']])
            processed_dataset['test'] = self.merge_dataset_splits([dataset['test_matched'],
                                                                  dataset['test_mismatched']])
            return processed_dataset

        if dataset_name == 'news-category-hf':
            dataset = load_dataset('Fraser/news-category-dataset', split=self.split, cache_dir=self.hf_cache_dir)
            merged_dataset = self.merge_dataset_splits([dataset['train'], dataset['test'], dataset['validation']])
            # self.id_classes = data_util.get_top_k_classes(dataset, k=5)
            self.id_classes = [0, 1, 2, 3, 4] # Original split from benchmark paper
            return merged_dataset

        if dataset_name == 'news-category-modified':
            self.label_key_name = self.dataset_to_keys[self.dataset_name][-1]
            self.id_classes = [0, 15, 24, 28, 18, 20, 21, 9, 31, 33, 34, 7, 12, 17, 38, 25, 22]
            self.class_mappings_modified_to_original = {0:0, 1:15, 2:24, 3:28, 4:18, 5:20, 6:21, 7:9, 8:31,   # ID Classes
                                                        9:33, 10:34, 11:7, 12:12, 13:17, 14:38, 15:25, 16:22, # ID Classes
                                                        17:1, 18:2, 19:3, 20:4, 21:5, 22:6, 23:8, 24:10, 25:11, # OOD Classes
                                                        26:13, 27:14, 28:16, 29:19, 30:23, 31:26, 32:27, 33:29, # OOD Classes
                                                        34:30, 35:32, 36:35, 37:36, 38:37, 39:39, 40:40}        # OOD Classes
            self.class_mappings_original_to_modified = {v: k for k, v in self.class_mappings_modified_to_original.items()}

            # dataset = load_dataset('Fraser/news-category-dataset', split=self.split, cache_dir=self.hf_cache_dir)
            # merged_dataset = self.merge_dataset_splits([dataset['train'], dataset['test'], dataset['validation']])
            #
            # # Change the class labels to be contiguous within ID and OOD and zero indexed
            # def change_label(example):
            #     example[self.label_key_name] = self.class_mappings_original_to_modified[example[self.label_key_name]]
            #     return example
            #
            # merged_dataset_modified = merged_dataset.map(change_label)
            # sampled_dataset = self.apply_sampling_class_imbalance(merged_dataset_modified, dataset_name=self.dataset_name)
            #
            # modified_id_classes = [self.class_mappings_original_to_modified[x] for x in self.id_classes]
            # modified_ood_classes = list(set(self.dataset_to_labels[dataset_name]) - set(modified_id_classes))
            #
            # id_dataset = self.get_data_from_classes(dataset=merged_dataset_modified, list_of_classes=modified_id_classes)
            # ood_dataset = self.get_data_from_classes(dataset=merged_dataset_modified, list_of_classes=modified_ood_classes)
            # id_dataset_sampled = self.get_data_from_classes(dataset=sampled_dataset, list_of_classes=modified_id_classes)
            # ood_dataset_sampled = self.get_data_from_classes(dataset=sampled_dataset, list_of_classes=modified_ood_classes)
            #
            # id_dataset = id_dataset.shuffle()
            # ood_dataset = ood_dataset.shuffle()
            # id_dataset_sampled = id_dataset_sampled.shuffle()
            # ood_dataset_sampled = ood_dataset_sampled.shuffle()
            #
            # # Ref: https://discuss.huggingface.co/t/how-to-use-a-nested-python-dictionary-in-dataset-from-dict/5757/6
            # def get_subset(data, low, high):
            #     return Dataset.from_dict(data[int(len(data) * low): int(len(data) * high)])
            #
            # processed_dataset = DatasetDict()
            # processed_dataset['id_train'] = get_subset(id_dataset, low=0, high=0.8)
            # processed_dataset['id_val'] = get_subset(id_dataset, low=0.8, high=0.85)
            # processed_dataset['id_test'] = get_subset(id_dataset, low=0.85, high=1)
            # processed_dataset['ood_train'] = get_subset(ood_dataset, low=0, high=0.8)
            # processed_dataset['ood_val'] = get_subset(ood_dataset, low=0.8, high=0.85)
            # processed_dataset['ood_test'] = get_subset(ood_dataset, low=0.85, high=1)
            # processed_dataset['id_train_sampled'] = get_subset(id_dataset_sampled, low=0, high=0.8)
            # processed_dataset['id_val_sampled'] = get_subset(id_dataset_sampled, low=0.8, high=0.85)
            # processed_dataset['id_test_sampled'] = get_subset(id_dataset_sampled, low=0.85, high=1)
            # processed_dataset['ood_train_sampled'] = get_subset(ood_dataset_sampled, low=0, high=0.8)
            # processed_dataset['ood_val_sampled'] = get_subset(ood_dataset_sampled, low=0.8, high=0.85)
            # processed_dataset['ood_test_sampled'] = get_subset(ood_dataset_sampled, low=0.85, high=1)
            # processed_dataset.save_to_disk(os.path.join(DATA_DIR, 'news_category_modified'))
            #
            # for split_name in processed_dataset:
            #     logger.info(f'Classes in {split_name} : {set(processed_dataset[split_name][self.label_key_name])}')

            '''
                        Note on NewsCategory:
                        Example datum: {'category_num': 0, 'category': 'POLITICS', 'headline': "Will Meanness Win The Day? If So, It'll Happen Quickly", 'authors': 'Chris Weigant, ContributorChris Weigant is a political commentator.', 'link': 'https://www.huffingtonpost.com/entry/will-meanness-win-the-day-if-so-itll-happen-quickly_us_59519c30e4b0326c0a8d0ac5', 'short_description': 'Either the bill fails because at least five or six Republicans declare their opposition, or the entire thing will pass with blinding speed.', 'date': '2017-06-26'}
                        There are three fields of note: 'headline', 'short_description' and 'link'. 'short_description' is sometimes an empty string.
                        The OOD benchmark paper implementation doesn't include this dataset. The only implementation I found online combines the headline and short_description categories into a single field.
                        That's what I'm going to do for now. Can switch to other data representation (like scraping full article from internet) if needed later. (But short description seem good for me to identify the label, so hopefully it's enough for the model too.)
    
                        -----------------------------
    
                        Original: Train /Test/ Val = 160682/ 30128/ 10043 = 80% / 15% / 5%
    
                        -----------------------------
    
                        Division of ID and OOD classes:
                        Original: ID/OOD = 86160 / 114693 -> 43% is ID
                        Modified: ID/OOD = 81968 / 118885-> 41% is ID
    
                        Modified classes:
                        ID Classes:
    
                            Class ID: 0, Class: POLITICS, Frequency: 32739
                            Class ID: 15, Class: THE WORLDPOST, Frequency: 3664
                            Class ID: 24, Class: WORLDPOST, Frequency: 2579
                            Class ID: 28, Class: WORLD NEWS, Frequency: 2177
    
                            Class ID: 18, Class: IMPACT, Frequency: 3459
    
                            Class ID: 20, Class: CRIME, Frequency: 3405
    
                            Class ID: 21, Class: MEDIA, Frequency: 2815
    
                            Class ID: 9, Class: BUSINESS, Frequency: 5937
                            Class ID: 31, Class: MONEY, Frequency: 1707
    
                            Class ID: 33, Class: FIFTY, Frequency: 1401
    
                            Class ID: 34, Class: GOOD NEWS, Frequency: 1398
    
                            Class ID: 7, Class: QUEER VOICES, Frequency: 6314
                            Class ID: 12, Class: BLACK VOICES, Frequency: 4528
                            Class ID: 17, Class: WOMEN, Frequency: 3490
                            Class ID: 38, Class: LATINO VOICES, Frequency: 1129
    
                            Class ID: 25, Class: RELIGION, Frequency: 2556
    
                            Class ID: 22, Class: WEIRD NEWS, Frequency: 2670
    
    
                        OOD Classes:
    
                            Class ID: 4, Class: STYLE & BEAUTY, Frequency: 9649
                            Class ID: 26, Class: STYLE, Frequency: 2254
    
                            Class ID: 39, Class: CULTURE & ARTS, Frequency: 1030
                            Class ID: 35, Class: ARTS & CULTURE, Frequency: 1339
                            Class ID: 32, Class: ARTS, Frequency: 1509
    
                            Class ID: 8, Class: FOOD & DRINK, Frequency: 6226
                            Class ID: 29, Class: TASTE, Frequency: 2096
    
                            Class ID: 37, Class: COLLEGE, Frequency: 1144
                            Class ID: 40, Class: EDUCATION, Frequency: 1004
    
                            Class ID: 27, Class: SCIENCE, Frequency: 2178
                            Class ID: 30, Class: TECH, Frequency: 2082
    
                            Class ID: 11, Class: SPORTS, Frequency: 4884
    
                            Class ID: 1, Class: WELLNESS, Frequency: 17827
                            Class ID: 6, Class: HEALTHY LIVING, Frequency: 6694
                            Class ID: 13, Class: HOME & LIVING, Frequency: 4195
                            Class ID: 3, Class: TRAVEL, Frequency: 9887
    
                            Class ID: 5, Class: PARENTING, Frequency: 8677
                            Class ID: 14, Class: PARENTS, Frequency: 3955
                            Class ID: 16, Class: WEDDINGS, Frequency: 3651
                            Class ID: 19, Class: DIVORCE, Frequency: 3426
    
                            Class ID: 2, Class: ENTERTAINMENT, Frequency: 16058
                            Class ID: 10, Class: COMEDY, Frequency: 5175
    
                            Class ID: 36, Class: ENVIRONMENT, Frequency: 1323
                            Class ID: 23, Class: GREEN, Frequency: 2622
                        '''

            processed_dataset = load_from_disk(os.path.join(paths.DATA_DIR, 'news_category_modified'))
            return processed_dataset

        if dataset_name == 'news-category-json':
            return load_dataset('json', data_files=paths.DATA_DIR+'/News_Category_Dataset_v2.json')

        if dataset_name == 'dbpedia-local':
            def correct_label(example):
                example['label'] = example['label'] - 1
                return example

            dataset = self.merge_dataset_splits(
                [load_dataset('csv', data_files=paths.DATA_DIR + '/dbpedia_csv/train.csv')['train'],
                 load_dataset('csv', data_files=paths.DATA_DIR + '/dbpedia_csv/test.csv')['train']])

            updated_dataset = dataset.map(correct_label)
            self.id_classes = [0, 1, 2, 3]
            return updated_dataset

        if dataset_name == 'clinc150':
            # # First time processing
            # raw_data = json.load(open(os.path.join(paths.DATA_DIR, dataset_name, 'data_full.json'), 'r'))
            #
            # labels = []
            # for split_name, split in raw_data.items():
            #     labels.extend([x[1] for x in split])
            # labels = list(set(labels))
            # labels.remove('oos')
            #
            # labels_to_label_indices = {x:i for i, x in enumerate(labels)}
            # labels_to_label_indices['oos'] = len(labels_to_label_indices) # 'oos' is 150
            # json.dump(labels_to_label_indices, open(os.path.join(paths.DATA_DIR, dataset_name, 'labels_to_label_indices.json'), "w"), indent = 4)
            #
            # processed_dataset = DatasetDict()
            # for k, v in raw_data.items():
            #     data_dict = {'sentence': [x[0] for x in v],
            #                  'label_text': [x[1] for x in v],
            #                  'label': [labels_to_label_indices[x[1]] for x in v]}
            #     processed_dataset[k] = Dataset.from_dict(data_dict)
            # processed_dataset.save_to_disk(os.path.join(paths.DATA_DIR, dataset_name, 'clinc150_processed'))

            # Routine loading
            self.labels_to_label_indices = json.load(open(os.path.join(paths.DATA_DIR, dataset_name, 'labels_to_label_indices.json'), "r"))
            self.label_indices_to_labels = {int(v): k for k, v in self.labels_to_label_indices.items()}
            self.id_classes = list(range(150))
            return load_from_disk(os.path.join(paths.DATA_DIR, dataset_name, 'clinc150_processed'))

        if dataset_name == '20newsgroups':
            from sklearn.datasets import fetch_20newsgroups
            processed_dataset = DatasetDict()

            for split in ['train', 'test']:  # splits are 'train', 'test' or 'all'
                raw_dataset = fetch_20newsgroups(subset=split)
                dataset_split = []
                for i in range(len(raw_dataset.data)):
                    sentence = raw_dataset.data[i]
                    label = raw_dataset.target[i]
                    label_text = raw_dataset.target_names[label]
                    dataset_split.append({'sentence': sentence, 'label': label, 'label_text': label_text})

                if split == 'test':
                    processed_dataset['val'] = Dataset.from_dict({
                        'sentence': [x['sentence'] for x in dataset_split[:2000]],
                        'label': [x['label'] for x in dataset_split[:2000]],
                        'label_text': [x['label_text'] for x in dataset_split[:2000]]})
                    processed_dataset['test'] = Dataset.from_dict({
                        'sentence': [x['sentence'] for x in dataset_split[2000:]],
                        'label': [x['label'] for x in dataset_split[2000:]],
                        'label_text': [x['label_text'] for x in dataset_split[2000:]]})
                else:
                    processed_dataset[split] = Dataset.from_dict({
                        'sentence': [x['sentence'] for x in dataset_split],
                        'label': [x['label'] for x in dataset_split],
                        'label_text': [x['label_text'] for x in dataset_split]
                })
            return processed_dataset

        if dataset_name == '20newsgroups-open-set-classification':
            self.id_classes = [1, 2, 3, 6, 7, 8, 11, 12, 16, 18, 19]
            self.class_mappings_modified_to_original = {0:1, 1:2, 2:3, 3:6, 4:7, 5:8, 6:11, 7:12, 8:16, 9:18, 10:19,  # ID Classes
                                                        11:0, 12:4, 13:5, 14:9, 15:10, 16:13, 17:14, 18:15, 19:17}    # OOD Classes
            self.class_mappings_original_to_modified = {v: k for k, v in self.class_mappings_modified_to_original.items()}

            # Ref:  http://qwone.com/~jason/20Newsgroups/
            #       https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
            #       https://www.kaggle.com/datasets/crawford/20-newsgroups
            #       https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
            #       https://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html
            # Preprocessing: https://github.com/filipefilardi/text-classification/blob/main/notebooks/preprocesing.ipynb

            # # First time processing
            # import re
            # from sklearn.datasets import fetch_20newsgroups
            #
            # def clean_text(text):
            #     # Header
            #     text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
            #     text = re.sub(r'(Subject:[^\n]+\n)', '', text)
            #     text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
            #     text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
            #     text = re.sub(r'(Version:[^\n]+\n)', '', text)
            #
            #     # Main Text
            #     re_url = re.compile(
            #         r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
            #     re_email = re.compile(
            #         '(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
            #
            #     text = text.strip()
            #     text = re.sub(re_url, '', text)   # Removes URLs
            #     text = re.sub(re_email, '', text) # Removes emails
            #
            #     text = re.sub(r'(\d+)', ' ', text) # Remove digits
            #     text = re.sub(r'(\s+)', ' ', text) # Remove whitespace like \n
            #     return text
            #
            # raw_dataset = fetch_20newsgroups(subset='all') # ‘train’, ‘test’, ‘all’
            # processed_dataset = DatasetDict()
            #
            # id_dataset, ood_dataset = [], []
            # for i in range(len(raw_dataset.data)):
            #     sentence = clean_text(raw_dataset.data[i])
            #     original_label = raw_dataset.target[i]
            #     modified_label = self.class_mappings_original_to_modified[original_label]
            #     label_text = raw_dataset.target_names[original_label]
            #     if original_label in self.id_classes:
            #         id_dataset.append({'sentence': sentence, 'label': modified_label, 'label_text': label_text})
            #     else:
            #         ood_dataset.append({'sentence': sentence, 'label': modified_label, 'label_text': label_text})
            #
            # import random
            # random.shuffle(id_dataset)
            # id_train_dataset = id_dataset[:int(0.85 * len(id_dataset))]
            # id_test_dataset = id_dataset[int(0.85 * len(id_dataset)):]
            #
            # processed_dataset['id_train'] = Dataset.from_dict({
            #     'sentence': [x['sentence'] for x in id_train_dataset],
            #     'label': [x['label'] for x in id_train_dataset],
            #     'label_text': [x['label_text'] for x in id_train_dataset]
            # })
            # processed_dataset['id_test'] = Dataset.from_dict({
            #     'sentence': [x['sentence'] for x in id_test_dataset],
            #     'label': [x['label'] for x in id_test_dataset],
            #     'label_text': [x['label_text'] for x in id_test_dataset]
            # })
            # processed_dataset['ood'] = Dataset.from_dict({
            #     'sentence': [x['sentence'] for x in ood_dataset],
            #     'label': [x['label'] for x in ood_dataset],
            #     'label_text': [x['label_text'] for x in ood_dataset]
            # })
            # processed_dataset.save_to_disk(os.path.join(paths.DATA_DIR, dataset_name, '20newsgroups_processed'))
            #
            # labels_to_label_indices = {i: raw_dataset.target_names[self.class_mappings_modified_to_original[i]]
            #                            for i in range(len(raw_dataset.target_names))}
            # json.dump(labels_to_label_indices,
            #           open(os.path.join(paths.DATA_DIR, dataset_name, 'labels_to_label_indices.json'), "w"), indent=4)

            # Routine loading
            self.label_indices_to_labels = json.load(open(os.path.join(paths.DATA_DIR, dataset_name, 'labels_to_label_indices.json'), "r"))
            self.labels_to_label_indices = {v: int(k) for k, v in self.label_indices_to_labels.items()}
            return load_from_disk(os.path.join(paths.DATA_DIR, dataset_name, '20newsgroups_processed'))

        if dataset_name == 'multi30k':
            # Get multi30k data from https://github.com/multi30k/dataset/tree/master/data/task1/tok
            # In the Contrastive paper, we use the union of test_2016_flickr.en, test_2017_mscoco.en, and test_2018_flickr.en

            lines = []
            for file_name in os.listdir(paths.DATA_DIR + '/multi30k'):
                with open(paths.DATA_DIR + '/multi30k/' + file_name, 'r') as f:
                    lines.extend([line.strip() for line in f])

            processed_dataset = DatasetDict()
            processed_dataset['test'] = Dataset.from_dict({
                'sentence': [x for x in lines],
                'label': [0] * len(lines)})
            return processed_dataset


        # dbpedia_14 - Error in downloading. Use 'dbpedia-local' instead.

        # 'token_type_ids' - Segment token indices to indicate first and second portions of the inputs.
        # Note: If dealing with seq2seq data, ensure that for lm_labels, PAD tokens are set as -100 to be ignored in loss calculation
        # dataset['text' / 'label' / 'input_ids' / 'token_type_ids' / 'attention_mask']
        # next(iter(dataloader)) gives a dict {'label', 'input_ids', 'token_type_ids', 'attention_mask'} -> All (1, 512) torch tensors.


    def add_dataset_metadata(self):
        '''
        Creates data structures containing meta-data on all datasets we will be using.
        :return:
        '''

        # The first two keys are input sentences; the last key is the label.
        self.dataset_to_keys = {
            'yelp_polarity': ('text', None, 'label'),
            'imdb': ('text', None, 'label'),
            'sst2': ("sentence", None, 'idx', 'label'),
            'news-category-hf': ('headline', 'short_description', 'category', 'authors', 'link', 'date', 'category_num'),
            'news-category-modified': ('headline', 'short_description', 'category', 'authors', 'link', 'date', 'category_num'),
            'dbpedia-local': ('title', 'content', 'label'),
            'mnli': ('premise', 'hypothesis', 'label', 'idx'),
            'rte': ('sentence1', 'sentence2', 'idx', 'label'),
            'clinc150': ('sentence', None, 'label_text', 'label'),
            '20newsgroups': ('sentence', None, 'label_text', 'label'),
            '20newsgroups-open-set-classification': ('sentence', None, 'label_text', 'label'),
            'multi30k': ('sentence', None, 'label')
            }

        self.dataset_to_labels = {'yelp_polarity': [0, 1],
                                  'imdb': [0, 1],
                                  'sst2': [0, 1], # -1 for hidden test split
                                  'news-category-hf': [x for x in range(41)],
                                  'news-category-modified': [x for x in range(41)],
                                  'dbpedia-local': [x for x in range(14)],
                                  'mnli': [0, 1, 2], # -1 for hidden test split
                                  'rte': [0, 1], # -1 for hidden test split
                                  'clinc150': [x for x in range(151)],
                                  '20newsgroups': [x for x in range(20)],
                                  '20newsgroups-open-set-classification': [x for x in range(20)],
                                  'multi30k': [0], # Unlabelled
                                  }

        self.dataset_to_split = {

            # Bad setup with these datasets
            'yelp_polarity': ['train', 'test'],
            'dbpedia-local': ['train'],
            'news-category-hf': ['train', 'validation', 'test'],
            '20newsgroups-open-set-classification': ['id_train', 'id_test', 'ood'],

            # [ID train, ID test, ID Val (for training)]
            'imdb': ['train', 'unsupervised', 'test'], # 'unsupervised' is unlabelled
            'sst2': ['train', 'test', 'validation'],   # 'test' is unlabelled
            'mnli': ['train', 'test', 'val'],          # 'test' is unlabelled
            'rte': ['train', 'test', 'validation'],    # 'test' is unlabelled
            '20newsgroups': ['train', 'test', 'val'],
            'multi30k': [None, 'test', None],          # Used only as OOD, so only one unlabelled test split

            # Semantic Shift only - [ID train, OOD test, ID val, ID test, ...]
            'news-category-modified': ['id_train_sampled', 'ood_test_sampled', 'id_val_sampled', 'id_test_sampled',
                                       'id_train', 'id_val', 'id_test', 'ood_train', 'ood_val', 'ood_test',
                                       'ood_train_sampled', 'ood_val_sampled'],
            'clinc150': ['train', 'oos_test', 'val', 'test', 'oos_train', 'oos_val'],
        }


    def preprocess_function(self, examples):
        # For RoBERTa, 'Ġ' means the end of a new token, and it appears at the start of most tokens.
        sentence1_key, sentence2_key = self.dataset_to_keys[self.dataset_name][:2]
        if sentence2_key is None:
            return self.tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=self.max_length)
        return self.tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=self.max_length) # Tokenized as <s> seq1 </s><s> seq1 </s>


    def get_tensors_end_to_end(self, split):
        '''
        Loads a dataset by name and applies pre-processing on it.
        :param split: Which split to process. If None, all splits are used.
        :return:
        '''

        dataset = self.get_dataset(self.dataset_name, split=split)
        logger.info(f'Dataset {self.dataset_name} loaded. Starting processing...')
        dataset = self.get_tensors_for_finetuning(dataset)
        logger.info('Data processing complete.')
        return dataset


    def get_tensors_for_finetuning(self, dataset, format='torch'):
        '''
        Applies pre-processing on a dataset object and computes tensors.
        :param dataset: Single split dataset.
        :return:
        '''
        dataset = dataset.map(self.preprocess_function, batched=True)

        if 'label' not in dataset.features:
            dataset = dataset.rename_column(self.dataset_to_keys[self.dataset_name][-1], 'label')

        dataset.set_format(type=format, columns=['input_ids', 'attention_mask', 'label'])
        return dataset


    def get_tensors_for_mlm(self, dataset, format='torch'):
        '''
        Applies pre-processing on a dataset object and computes tensors.
        :param dataset: Single split dataset.
        :return:
        '''

        def tokenize_function(examples):
            return self.tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=self.max_length)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        dataset = dataset.map(tokenize_function, batched=True)

        samples = dataset[:]['input_ids']
        masked_inputs = data_collator(samples)
        dataset = dataset.remove_columns(['input_ids', 'label', 'sentence', 'label_text'])
        dataset = dataset.add_column('input_ids', masked_inputs['input_ids'].tolist())
        dataset = dataset.add_column('label', masked_inputs['labels'].tolist())

        dataset.set_format(type=format, columns=['input_ids', 'attention_mask', 'label'])

        return dataset


    def get_dataloader(self, dataset, batch_size=1):
        '''
        Given a dataset object, converts it into a DataLoader object.
        :param dataset:
        :param batch_size:
        :return:
        '''
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


    def merge_dataset_splits(self, dataset_split_list):
        '''
        Given a list of datasets representing different splits of a dataset, returns a single dataset item.
        :param dataset_split_list: List of single split dataset objects.
        :return:
        '''
        return concatenate_datasets(dataset_split_list)


    def get_data_from_classes(self, dataset, list_of_classes):
        '''
        From a dataset object containing a single split, keeps data from a specified list of classes.
        :param dataset: Single split dataset.
        :param list_of_classes:
        :return: Single split dataset.
        '''
        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        filtered_dataset = dataset.filter(lambda example: example[label_key_name] in list_of_classes)
        if not self.debug_mode:
            assert sorted(list_of_classes) == sorted(list(self.get_unique_labels(filtered_dataset)))
        return filtered_dataset


    def get_unique_labels(self, dataset):
        '''
        Given a dataset object containing a single split, returns the unique labels in it.
        :param dataset: Single split dataset.
        :return:
        '''
        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        return set([dataset[i][label_key_name] for i in range(len(dataset))])


    def apply_sampling_class_imbalance(self, dataset, dataset_name):

        def set_sampling_probs(counter, coeff=0.5):
            """
            Set the probability of sampling specific languages / language pairs during training.
            Ref: https://github.com/facebookresearch/XLM/blob/main/xlm/utils.py#L195
            """
            assert coeff > 0

            probs = np.array([x[1] for x in counter])
            probs = probs.astype('float32') / probs.sum()
            probs = np.array([p ** coeff for p in probs])
            probs /= probs.sum()

            return probs

        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        class_counts = self.get_class_frequencies(dataset, dataset_name)
        num_samples, num_classes = sum([x[1] for x in class_counts]), len(class_counts)
        sampling_probs = set_sampling_probs(class_counts, coeff=0.5)

        draw = np.random.choice([x[0] for x in class_counts], num_samples, p=sampling_probs)
        sampled_class_counts = {}
        for k in range(num_classes):
            sampled_class_counts[k] = sum(draw == k)
        sampled_class_counts = {k: sum(draw == k) for k in range(len(class_counts))}

        samples_to_keep = []
        for class_idx in range(num_classes):
            # if i < 1: # TODO: Most frequent class - keep original counts
            #     pass
            # else:    # Upsample
            original_class_samples = self.get_data_from_classes(dataset, [class_idx])
            indicies_to_sample = np.random.choice(len(original_class_samples), size=sampled_class_counts[class_idx], replace=True)
            samples_to_keep.append(Dataset.from_dict(original_class_samples[indicies_to_sample]))

        sampled_dataset = self.merge_dataset_splits(samples_to_keep)

        class_counts = {x[0]:x[1] for x in class_counts}
        print(f'Original: \n{class_counts}')
        print(f'Modified: \n{sampled_class_counts}')

        plt.bar(x=class_counts.keys(), height=class_counts.values(), label='Original', alpha=0.5)
        plt.bar(x=sampled_class_counts.keys(), height=sampled_class_counts.values(), label='Sampled', alpha=0.5)
        plt.title(f'Class Frequencies after Sampling - {dataset_name}')
        plt.grid()
        plt.legend()
        plt.savefig(f'plots/Class Frequencies after Sampling - {dataset_name}')
        plt.show()

        return sampled_dataset


    def get_class_frequencies(self, dataset, dataset_name):
        '''
        Given a dataset object containing a single split, calculates the class frequencies.
        :param dataset: Single split dataset.
        :return: [(id_1, freq_1)...(id_n, freq_n)]
        '''

        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        counter = collections.Counter([dataset[i][label_key_name] for i in range(len(dataset))])
        counter = sorted(counter.items(), key=lambda item: item[0])
        return counter


    def get_class_frequency_histogram(self, dataset, dataset_name, mark_id_ood=False):
        '''
        Given a dataset object containing a single split, plots the class frequencies.
        :param dataset: Single split dataset.
        :return:
        '''

        counter = self.get_class_frequencies(dataset, dataset_name)
        logger.info(f'Class frequencies: \n{counter}')

        if dataset_name == 'clinc150' or dataset_name == 'CLINC150':
            self.id_classes = [x for x in range(150)]

        # if dataset_name == 'NewsCategory':
        #     id_ood_class_boundary = 5
        # elif dataset_name == 'DBPedia Ontology':
        #     id_ood_class_boundary = 4
        # color=[colors['ID']]*id_ood_class_boundary + [colors['OOD']]*(len(counter)-id_ood_class_boundary),

        colours = {'ID': 'red', 'OOD': 'blue'}

        if mark_id_ood:
            colours_list = [colours['ID'] if x in self.id_classes else colours['OOD'] for x in range(len(counter))]
            colours_dict = colours
        else:
            colours_list = None
            colours_dict = None

        plot_bar(x=[x[0] for x in counter], y=[x[1] for x in counter],
                 x_label='Classes', y_label='Frequency',
                 title=f'Class Frequencies: {dataset_name}',
                 savename=f'{paths.output_plot_dir}/class-frequencies_{dataset_name}.png',
                 colours_list=colours_list,
                 colours_dict=colours_dict)


    def get_top_k_classes(self, dataset, k, show_class_mapping=False):
        '''
        Given a dataset object containing a single split, returns the names of the k most frequent classes.
        :param dataset: Single split dataset.
        :param k:
        :return: List of class indices
        '''

        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        counter = collections.Counter([dataset[i][label_key_name] for i in range(len(dataset))])
        class_counts = sorted(counter.items(), reverse=True, key=lambda item: item[1])
        top_k = class_counts[:k]
        logger.info(f'The following classes have the highest frequencies: \n{top_k}')

        if show_class_mapping:
            class_mapping = {}
            num_labels = len(set(dataset['category_num']))
            for datum in dataset:
                class_mapping[datum['category_num']] = datum['category']
                if len(class_mapping) == num_labels:
                    break
            class_mapping = sorted(class_mapping.items(), key=lambda item: item[0])
            class_counts = [(class_mapping[i], class_counts[i][1]) for i in range(len(class_counts))]
            for datum in class_counts:
                logger.info(f'Class ID: {datum[0][0]}, Class: {datum[0][1]}, Frequency: {datum[1]}')

        return [x[0] for x in top_k]


    def analyse_sequence_lengths(self, dataset, dataset_name):

        if self.dataset_name == 'dbpedia-local':
            seq_len = [len(x.split()) for x in dataset['content']]
        elif self.dataset_name == 'news-category-hf':
            seq_len = [len(f'{x["headline"]} {x["short_description"]}'.split()) for x in dataset]
        elif dataset_name in ['clinc150', 'CLINC150', '20 Newsgroups']:
            seq_len = [len(x.split()) for x in dataset['sentence']]

        seq_len = sorted(seq_len)
        logger.info(f'From {len(seq_len)} samples, \nAverage sequence length = {round(np.average(seq_len), 2)} words\n'
                    f'Max sequence length = {max(seq_len)} words\n'
                    f'Median sequence length = {np.median(seq_len)} words\n'
                    f'Sequence length at 95th percentile = {seq_len[int(len(seq_len) * 0.95)]}')

        plt.plot(seq_len)
        plt.xlabel('Datapoints')
        plt.ylabel('Number of whitespace seperated tokens')
        plt.title(f'{dataset_name} - Sequence Lengths')
        plt.grid()
        plt.text(int(len(seq_len) * 0.95), seq_len[int(len(seq_len) * 0.95)], f'Len = {seq_len[int(len(seq_len) * 0.95)]}')
        plt.savefig(f'{paths.output_plot_dir}/{dataset_name}-sequence-lengths.png')
        plt.show()



if __name__ == '__main__':

    ###################### DBPedia Ontology ##############################

    # dataset_name = 'dbpedia-local'
    # data_util = DatasetUtil(dataset_name=dataset_name)
    # dataset = data_util.get_dataset(dataset_name, split=None)
    # # processed_dataset = data_util.get_tensors(dataset)

    # list_of_id_classes = [0,1,2,3]   # Selection made on ascending class ID
    # id_dataset = data_util.get_data_from_classes(dataset=dataset, list_of_classes=list_of_id_classes)
    # ood_dataset = data_util.get_data_from_classes(dataset=dataset, list_of_classes=list(set(data_util.dataset_to_labels[dataset_name]) - set(list_of_id_classes)))

    # # Analysis
    # data_util.get_class_frequency_histogram(dataset, dataset_name='DBPedia Ontology')
    # data_util.analyse_sequence_lengths(dataset, dataset_name='DBPedia')
    # data_util.analyse_sequence_lengths(id_dataset, dataset_name='DBPedia (ID)')
    # data_util.analyse_sequence_lengths(ood_dataset, dataset_name='DBPedia (OOD)')

    ###################### NewsCategory  ##############################

    # dataset_name = 'news-category-modified'
    # data_util = DatasetUtil(dataset_name=dataset_name)
    # dataset = data_util.get_dataset(dataset_name, split=None)

    # Analysis
    # data_util.get_class_frequency_histogram(data_util.merge_dataset_splits([dataset[x] for x in dataset]), dataset_name='NewsCategory')
    # top_k_classes = data_util.get_top_k_classes(dataset, k=5)
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset[x] for x in dataset]), dataset_name='NewsCategory')
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset['id_train'], dataset['id_val'], dataset['id_test']]),
    #                                    dataset_name='NewsCategory (ID)')
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset['ood_train'], dataset['ood_val'], dataset['ood_test']]),
    #                                    dataset_name='NewsCategory (OOD)')

    ###################### CLINC150  ##############################

    # dataset_name = 'clinc150'
    # data_util = DatasetUtil(dataset_name=dataset_name)
    # dataset = data_util.get_dataset(dataset_name, split=None)

    # data_util.get_class_frequency_histogram(data_util.merge_dataset_splits([dataset[x] for x in dataset]),
    #                                         dataset_name='CLINC150')
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset['train'], dataset['val'], dataset['test']]),
    #                                    dataset_name='CLINC150')
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset['oos_train'],
    #                                    dataset['oos_val'], dataset['oos_test']]), dataset_name='CLINC150')

    ###################### 20 Newsgroups(open set classification) #####################################

    # dataset_name = '20newsgroups-open-set-classification'
    # data_util = DatasetUtil(dataset_name=dataset_name)
    # dataset = data_util.get_dataset(dataset_name, split=None)
    # data_util.get_class_frequency_histogram(data_util.merge_dataset_splits([dataset[x] for x in dataset]),
    #                                         dataset_name='20 Newsgroups')
    # data_util.analyse_sequence_lengths(data_util.merge_dataset_splits([dataset['id_train'], dataset['id_test']]),
    #                                    dataset_name='20 Newsgroups')
    # data_util.analyse_sequence_lengths(dataset['ood'], dataset_name='20 Newsgroups')

    ###########################################################

    dataset_name = '20newsgroups'
    # dataset_name = 'multi30k'
    data_util = DatasetUtil(dataset_name=dataset_name)
    dataset = data_util.get_dataset(dataset_name, split=None)
    # data_util.get_class_frequency_histogram(dataset['test'], dataset_name)
    print('Done.')

    ###############################################################
