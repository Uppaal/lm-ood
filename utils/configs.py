import os
import yaml


class Config(object):
    def __init__(self):

        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        config_file = os.path.join(self.root_path, 'configs.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError('configs.yaml is empty')
        self.update_kwargs(self.flatten_sections(config))
        self.set_runtime_environment()

        import torch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_gpu = torch.cuda.device_count()


    def flatten_sections(self, config):
        flat_config = {}
        for section_name, section_values in config.items():
            if not isinstance(section_values, dict):
                raise ValueError(f'{section_name} must be a mapping in configs.yaml')
            for key, value in section_values.items():
                if key in flat_config:
                    raise ValueError(f'{key} is defined more than once in configs.yaml')
                flat_config[key] = value
        return flat_config


    def update_kwargs(self, kwargs):
        missing_keys = self.config_keys() - set(kwargs.keys())
        if missing_keys:
            raise ValueError(f'Missing config values in configs.yaml: {sorted(missing_keys)}')

        for (k, v) in kwargs.items():
            if k not in self.config_keys():
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

        if self.model_path is None or self.model_path == '':
            raise ValueError('model_path must be set in configs.yaml')
        self.MODEL_DIR = self.resolve_path(self.MODEL_DIR)
        self.DATA_DIR = self.resolve_path(self.DATA_DIR)
        self.temp_output_dir = self.resolve_path(self.temp_output_dir)
        self.model_path = self.resolve_path(self.model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'model_path does not exist: {self.model_path}')
        self.savedir = self.MODEL_DIR + f'/finetuned_models/{self.model_class}_base_{self.task_name}'


    def set_runtime_environment(self):
        if self.CUDA_VISIBLE_DEVICES is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.CUDA_VISIBLE_DEVICES)

        if self.set_seed:
            os.environ['PYTHONHASHSEED'] = str(self.seed)
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


    def config_keys(self):
        return {
            'MODEL_DIR',
            'DATA_DIR',
            'temp_output_dir',
            'set_seed',
            'seed',
            'CUDA_VISIBLE_DEVICES',
            'debug_mode',
            'model_class',
            'model_path',
            'max_seq_length',
            'sentence_embedding',
            'layer_representation_for_ood',
            'task_name',
            'ood_datasets',
            'samples_to_keep',
            'do_train',
            'num_train_epochs',
            'linear_probe',
            'epoch_wise_eval',
            'batch_size',
            'learning_rate',
            'adam_epsilon',
            'warmup_ratio',
            'weight_decay',
            'contrastive_weight',
            'cross_entropy_weight',
            'contrastive_loss',
            'tau',
            'report_all_metrics',
        }


    def resolve_path(self, path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.root_path, path))


    def values(self):
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")

