import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_utils import DatasetUtil
from utils.trainer_utils import get_trainer_pretraining
from utils.model_utils import load_model, load_tokenizer
from utils.startup import exp_configs, logger, print_summary

os.environ["WANDB_DISABLED"] = "true"

dataset_name = exp_configs.task_name
original_model_path = exp_configs.model_path
savedir = exp_configs.MODEL_DIR + f'/pretrained_models/{exp_configs.model_class}_base_{dataset_name}_tapt'
debug_mode = exp_configs.debug_mode


def print_model_save_details(save_path, dataset_name, training_method):
    print('Model saved.')
    print(f'  path: {save_path}')
    print(f'  training dataset: {dataset_name}')
    print(f'  training method: {training_method}')


tokenizer = load_tokenizer(model_path=original_model_path, model_str=exp_configs.model_class)
data_util = DatasetUtil(dataset_name=dataset_name, max_length=exp_configs.max_seq_length, tokenizer=tokenizer)

model = load_model(model_path=original_model_path, load_only_body=True)

dataset = data_util.get_dataset(dataset_name, split=None)
logger.info(f'Dataset {dataset_name} loaded. Starting pre-processing...')
train_split = dataset[data_util.dataset_to_split[dataset_name][0]]
train_dataset = train_split.map(data_util.preprocess_function, batched=True,
    remove_columns=[x for x in data_util.dataset_to_keys[data_util.dataset_name] if x is not None])

trainer = get_trainer_pretraining(dataset=train_dataset, model=model, tokenizer=tokenizer, batch_size=16,
                                  num_epochs=exp_configs.num_train_epochs,
                                  output_dir=f"{exp_configs.temp_output_dir}/pretrain_results_tapt-{dataset_name}",
                                  seed=exp_configs.seed)

logger.info('Starting tapt_training...\n\n\n')
result = trainer.train()
print('Training complete! Saving model...')
model.save_pretrained(savedir)
data_util.tokenizer.save_pretrained(savedir)
print_model_save_details(savedir, dataset_name, 'TAPT MLM')

print(result)
print_summary(result)
