import os
import ast
import json
import torch


class Config(object):
    def __init__(self):

        # Set paths
        if os.path.split(os.getcwd())[-1] == 'src':
            self.root_path = os.getcwd()
        else:
            self.root_path = os.path.split(os.getcwd())[0]
        self.MODEL_DIR = f'{self.root_path}/../models'
        self.DATA_DIR = f'{self.root_path}/../data'
        self.output_plot_dir = f'{self.root_path}/../plots'
        self.temp_output_dir = f'{self.root_path}/../temp_outputs'

        # Run configs
        self.set_seed = True
        self.seed = 42

        # Model configs
        self.model_class = 'roberta'  # 'roberta' 't5' 'gpt2'
        self.max_seq_length = 256
        self.sentence_embedding = 'average'  # 'average' 'cls'
        self.layer_representation_for_ood = 'classifier_input'  # 'penultimate_layer' 'classifier_input'

        # Dataset configs
        self.task_name = None  # '20newsgroups' 'imdb' 'news-category-modified'
        self.ood_datasets = ['sst2', '20newsgroups', 'rte', 'mnli', 'imdb', 'multi30k', 'news-category-modified', 'clinc150']
        self.samples_to_keep = None

        # Training configs
        self.do_train = False
        self.num_train_epochs = 10
        self.linear_probe = False
        self.epoch_wise_eval = False
        self.batch_size = 4
        self.learning_rate = 1e-5
        self.adam_epsilon = 1e-8
        self.warmup_ratio = 0.0
        self.weight_decay = 0.01

        # Training loss configs
        self.contrastive_weight = 2.0
        self.cross_entropy_weight = 1.0
        self.contrastive_loss = 'supcon'  # 'supcon' 'margin'
        self.tau = 0.3  # Temperature for SupCon

        self.report_all_metrics = False
        self.debug_mode = False

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_gpu = torch.cuda.device_count()
        self.CUDA_VISIBLE_DEVICES = None

        config_file = os.path.join(self.root_path, '..', 'configs.json')
        self.update_kwargs(json.load(open(config_file)), eval=False)


    def update_kwargs(self, kwargs, eval=True):
        for (k, v) in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

        self.model_name_or_path = self.MODEL_DIR + f'/pretrained_models/{self.model_class}_base'
        self.savedir = self.MODEL_DIR + f'/finetuned_models/{self.model_class}_base_{self.task_name}'


    def values(self):
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")

