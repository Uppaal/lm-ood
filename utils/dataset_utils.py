import os
import json
import torch
import collections
import numpy as np
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
from utils.startup import exp_configs, logger


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
            self.hf_cache_dir = f'{exp_configs.DATA_DIR}/huggingface_datasets'

        self.tokenizer = tokenizer
        self.add_dataset_metadata()


    def get_dataset(self, dataset_name, split):
        logger.info(f'Loading {split} split of {dataset_name} dataset...')

        if dataset_name in ['imdb']:
            return load_dataset(dataset_name, split=split, cache_dir=self.hf_cache_dir)

        if dataset_name in ['sst2', 'rte']:
            return load_dataset('glue', dataset_name, split=split, cache_dir=self.hf_cache_dir)

        if dataset_name == 'mnli':
            dataset = load_dataset('glue', dataset_name, split=split, cache_dir=self.hf_cache_dir)
            processed_dataset = DatasetDict()
            processed_dataset['train'] = dataset['train']
            processed_dataset['val'] = self.merge_dataset_splits([dataset['validation_matched'],
                                                                  dataset['validation_mismatched']])
            processed_dataset['test'] = self.merge_dataset_splits([dataset['test_matched'],
                                                                  dataset['test_mismatched']])
            return processed_dataset

        if dataset_name == 'news-category-modified':
            self.label_key_name = self.dataset_to_keys[self.dataset_name][-1]
            self.id_classes = [0, 15, 24, 28, 18, 20, 21, 9, 31, 33, 34, 7, 12, 17, 38, 25, 22]
            self.class_mappings_modified_to_original = {0:0, 1:15, 2:24, 3:28, 4:18, 5:20, 6:21, 7:9, 8:31,   # ID Classes
                                                        9:33, 10:34, 11:7, 12:12, 13:17, 14:38, 15:25, 16:22, # ID Classes
                                                        17:1, 18:2, 19:3, 20:4, 21:5, 22:6, 23:8, 24:10, 25:11, # OOD Classes
                                                        26:13, 27:14, 28:16, 29:19, 30:23, 31:26, 32:27, 33:29, # OOD Classes
                                                        34:30, 35:32, 36:35, 37:36, 38:37, 39:39, 40:40}        # OOD Classes
            self.class_mappings_original_to_modified = {v: k for k, v in self.class_mappings_modified_to_original.items()}
            return self._load_or_create_news_category_modified()

        if dataset_name == 'clinc150':
            self.id_classes = list(range(150))
            return self._load_or_create_clinc150()

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

        if dataset_name == 'multi30k':
            return self._load_or_create_multi30k()


    def _saved_dataset_exists(self, dataset_path):
        return os.path.isdir(dataset_path) and len(os.listdir(dataset_path)) > 0


    def _slice_dataset(self, dataset, low, high):
        start = int(len(dataset) * low)
        end = int(len(dataset) * high)
        return Dataset.from_dict(dataset[start:end])


    def _sample_class_imbalance(self, dataset, coeff=0.5):
        rng = np.random.default_rng(exp_configs.seed)
        label_key_name = self.dataset_to_keys[self.dataset_name][-1]
        class_counts = collections.Counter(dataset[label_key_name])
        class_counts = sorted(class_counts.items(), key=lambda item: item[0])

        labels = [label for label, _ in class_counts]
        counts = np.array([count for _, count in class_counts], dtype='float32')
        sampling_probs = counts / counts.sum()
        sampling_probs = sampling_probs ** coeff
        sampling_probs = sampling_probs / sampling_probs.sum()

        draw = rng.choice(labels, size=int(counts.sum()), p=sampling_probs)
        sampled_splits = []
        for label in labels:
            label_dataset = self.get_data_from_classes(dataset, [label])
            sample_count = int(np.sum(draw == label))
            sample_indices = rng.choice(len(label_dataset), size=sample_count, replace=True)
            sampled_splits.append(Dataset.from_dict(label_dataset[sample_indices.tolist()]))

        return self.merge_dataset_splits(sampled_splits)


    def _load_or_create_news_category_modified(self):
        processed_path = os.path.join(exp_configs.DATA_DIR, 'news_category_modified')
        if self._saved_dataset_exists(processed_path):
            return load_from_disk(processed_path)

        logger.info(f'Processed NewsCategory data not found at {processed_path}. Creating it now...')
        merged_dataset = self._load_news_category_raw_dataset()

        def change_label(example):
            label = int(example[self.label_key_name])
            example[self.label_key_name] = self.class_mappings_original_to_modified[label]
            return example

        merged_dataset = merged_dataset.map(change_label)
        sampled_dataset = self._sample_class_imbalance(merged_dataset)

        modified_id_classes = [self.class_mappings_original_to_modified[x] for x in self.id_classes]
        modified_ood_classes = list(set(self.dataset_to_labels[self.dataset_name]) - set(modified_id_classes))

        id_dataset = self.get_data_from_classes(merged_dataset, modified_id_classes).shuffle(seed=exp_configs.seed)
        ood_dataset = self.get_data_from_classes(merged_dataset, modified_ood_classes).shuffle(seed=exp_configs.seed)
        id_dataset_sampled = self.get_data_from_classes(sampled_dataset, modified_id_classes).shuffle(seed=exp_configs.seed)
        ood_dataset_sampled = self.get_data_from_classes(sampled_dataset, modified_ood_classes).shuffle(seed=exp_configs.seed)

        processed_dataset = DatasetDict()
        processed_dataset['id_train'] = self._slice_dataset(id_dataset, low=0, high=0.8)
        processed_dataset['id_val'] = self._slice_dataset(id_dataset, low=0.8, high=0.85)
        processed_dataset['id_test'] = self._slice_dataset(id_dataset, low=0.85, high=1)
        processed_dataset['ood_train'] = self._slice_dataset(ood_dataset, low=0, high=0.8)
        processed_dataset['ood_val'] = self._slice_dataset(ood_dataset, low=0.8, high=0.85)
        processed_dataset['ood_test'] = self._slice_dataset(ood_dataset, low=0.85, high=1)
        processed_dataset['id_train_sampled'] = self._slice_dataset(id_dataset_sampled, low=0, high=0.8)
        processed_dataset['id_val_sampled'] = self._slice_dataset(id_dataset_sampled, low=0.8, high=0.85)
        processed_dataset['id_test_sampled'] = self._slice_dataset(id_dataset_sampled, low=0.85, high=1)
        processed_dataset['ood_train_sampled'] = self._slice_dataset(ood_dataset_sampled, low=0, high=0.8)
        processed_dataset['ood_val_sampled'] = self._slice_dataset(ood_dataset_sampled, low=0.8, high=0.85)
        processed_dataset['ood_test_sampled'] = self._slice_dataset(ood_dataset_sampled, low=0.85, high=1)

        os.makedirs(exp_configs.DATA_DIR, exist_ok=True)
        processed_dataset.save_to_disk(processed_path)
        logger.info(f'Saved processed NewsCategory data to {processed_path}')
        return processed_dataset


    def _load_news_category_raw_dataset(self):
        raw_dataset = load_dataset('heegyu/news-category-dataset', cache_dir=self.hf_cache_dir)
        if isinstance(raw_dataset, DatasetDict):
            split_order = [name for name in ['train', 'test', 'validation'] if name in raw_dataset]
            split_order.extend([name for name in raw_dataset.keys() if name not in split_order])
            dataset = self.merge_dataset_splits([raw_dataset[name] for name in split_order])
        else:
            dataset = raw_dataset

        if self.label_key_name in dataset.column_names:
            return dataset
        return self._add_news_category_num(dataset)


    def _add_news_category_num(self, dataset):
        if 'category' not in dataset.column_names:
            raise ValueError('NewsCategory data must contain either category_num or category.')

        # The public source includes a newer U.S. NEWS category. The original setup uses 41 categories.
        dataset = dataset.filter(lambda example: example['category'] != 'U.S. NEWS')
        categories = sorted(set(dataset['category']))
        if len(categories) != 41:
            raise ValueError(f'Expected 41 NewsCategory classes, found {len(categories)}.')

        category_to_num = {category: i for i, category in enumerate(categories)}

        def add_label(example):
            example[self.label_key_name] = category_to_num[example['category']]
            return example

        return dataset.map(add_label)


    def _load_clinc150_labels(self, labels_path, processed_dataset=None):
        if os.path.exists(labels_path):
            self.labels_to_label_indices = json.load(open(labels_path, 'r'))
        elif processed_dataset is not None:
            labels = {}
            for split in processed_dataset.values():
                for label_text, label in zip(split['label_text'], split['label']):
                    labels[label_text] = int(label)
            self.labels_to_label_indices = labels
            json.dump(self.labels_to_label_indices, open(labels_path, 'w'), indent=4)
        else:
            raise FileNotFoundError(f'CLINC150 label mapping not found: {labels_path}')
        self.label_indices_to_labels = {int(v): k for k, v in self.labels_to_label_indices.items()}


    def _load_or_create_clinc150(self):
        dataset_dir = os.path.join(exp_configs.DATA_DIR, 'clinc150')
        processed_path = os.path.join(dataset_dir, 'clinc150_processed')
        labels_path = os.path.join(dataset_dir, 'labels_to_label_indices.json')

        if self._saved_dataset_exists(processed_path):
            processed_dataset = load_from_disk(processed_path)
            self._load_clinc150_labels(labels_path, processed_dataset=processed_dataset)
            return processed_dataset

        raw_path = os.path.join(dataset_dir, 'data_full.json')
        os.makedirs(dataset_dir, exist_ok=True)

        if os.path.exists(raw_path):
            logger.info(f'Processed CLINC150 data not found at {processed_path}. Creating it from {raw_path}...')
            raw_data = json.load(open(raw_path, 'r'))
            processed_dataset = self._build_clinc150_from_raw(raw_data)
        else:
            logger.info(f'CLINC150 raw data not found at {raw_path}. Downloading clinc_oos now...')
            processed_dataset = self._build_clinc150_from_hf()

        json.dump(self.labels_to_label_indices, open(labels_path, 'w'), indent=4)
        processed_dataset.save_to_disk(processed_path)
        logger.info(f'Saved processed CLINC150 data to {processed_path}')
        return processed_dataset


    def _build_clinc150_from_raw(self, raw_data):
        label_names = sorted({datum[1] for split in raw_data.values() for datum in split if datum[1] != 'oos'})
        self.labels_to_label_indices = {label: i for i, label in enumerate(label_names)}
        self.labels_to_label_indices['oos'] = len(self.labels_to_label_indices)
        self.label_indices_to_labels = {int(v): k for k, v in self.labels_to_label_indices.items()}

        processed_dataset = DatasetDict()
        for split_name, split_data in raw_data.items():
            processed_dataset[split_name] = Dataset.from_dict({
                'sentence': [datum[0] for datum in split_data],
                'label_text': [datum[1] for datum in split_data],
                'label': [self.labels_to_label_indices[datum[1]] for datum in split_data],
            })

        return processed_dataset


    def _build_clinc150_from_hf(self):
        raw_dataset = load_dataset('clinc_oos', 'plus', cache_dir=self.hf_cache_dir)
        label_names = raw_dataset['train'].features['intent'].names
        id_label_names = sorted([label for label in label_names if label != 'oos'])

        self.labels_to_label_indices = {label: i for i, label in enumerate(id_label_names)}
        self.labels_to_label_indices['oos'] = len(self.labels_to_label_indices)
        self.label_indices_to_labels = {int(v): k for k, v in self.labels_to_label_indices.items()}

        processed_dataset = DatasetDict()
        for source_split, id_split, oos_split in [
            ('train', 'train', 'oos_train'),
            ('validation', 'val', 'oos_val'),
            ('test', 'test', 'oos_test'),
        ]:
            id_rows, oos_rows = self._split_clinc150_hf_rows(raw_dataset[source_split], label_names)
            processed_dataset[id_split] = Dataset.from_dict(id_rows)
            processed_dataset[oos_split] = Dataset.from_dict(oos_rows)

        return processed_dataset


    def _split_clinc150_hf_rows(self, dataset, label_names):
        id_rows = {'sentence': [], 'label_text': [], 'label': []}
        oos_rows = {'sentence': [], 'label_text': [], 'label': []}

        for example in dataset:
            label_text = label_names[int(example['intent'])]
            rows = oos_rows if label_text == 'oos' else id_rows
            rows['sentence'].append(example['text'])
            rows['label_text'].append(label_text)
            rows['label'].append(self.labels_to_label_indices[label_text])

        return id_rows, oos_rows


    def _load_or_create_multi30k(self):
        dataset_dir = os.path.join(exp_configs.DATA_DIR, 'multi30k')
        processed_path = os.path.join(dataset_dir, 'multi30k_processed')

        if self._saved_dataset_exists(processed_path):
            return load_from_disk(processed_path)

        lines = self._load_multi30k_local_lines(dataset_dir)
        if not lines:
            logger.info(f'Multi30K data not found at {dataset_dir}. Downloading it now...')
            raw_dataset = load_dataset('bentrevett/multi30k', cache_dir=self.hf_cache_dir)
            lines = []
            for split in raw_dataset.values():
                lines.extend(split['en'])

        if not lines:
            raise ValueError('Multi30K dataset is empty after loading.')

        processed_dataset = DatasetDict()
        processed_dataset['test'] = Dataset.from_dict({
            'sentence': lines,
            'label': [0] * len(lines),
        })

        os.makedirs(dataset_dir, exist_ok=True)
        processed_dataset.save_to_disk(processed_path)
        logger.info(f'Saved processed Multi30K data to {processed_path}')
        return processed_dataset


    def _load_multi30k_local_lines(self, dataset_dir):
        if not os.path.isdir(dataset_dir):
            return []

        lines = []
        for file_name in sorted(os.listdir(dataset_dir)):
            file_path = os.path.join(dataset_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                lines.extend([line.strip() for line in f if line.strip()])
        return lines


    def add_dataset_metadata(self):
        '''
        Creates data structures containing meta-data on all datasets we will be using.
        :return:
        '''

        # The first two keys are input sentences; the last key is the label.
        self.dataset_to_keys = {
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

        self.dataset_to_labels = {'imdb': [0, 1],
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
            'dbpedia-local': ['train'],
            'news-category-hf': ['train', 'validation', 'test'],
            '20newsgroups-open-set-classification': ['id_train', 'id_test', 'ood'],

            # [ID train, ID test, ID Val (for tapt_training)]
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
        if not exp_configs.debug_mode:
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
