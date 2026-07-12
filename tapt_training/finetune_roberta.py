import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_utils import DatasetUtil
from utils.trainer_utils import get_trainer_finetuning
from utils.model_utils import load_model, load_tokenizer
from utils.startup import exp_configs, logger, print_summary


dataset_name = exp_configs.task_name
tapt_model_path = exp_configs.MODEL_DIR + f'/pretrained_models/{exp_configs.model_class}_base_{dataset_name}_tapt'
trained_model_path = f'/{exp_configs.model_class}_base_{dataset_name}'
savedir = exp_configs.savedir
debug_mode = exp_configs.debug_mode


def print_model_save_details(save_path, dataset_name, training_method):
    print('Model saved.')
    print(f'  path: {save_path}')
    print(f'  training dataset: {dataset_name}')
    print(f'  training method: {training_method}')


if not os.path.isdir(tapt_model_path):
    raise FileNotFoundError(
        f'TAPT checkpoint not found: {tapt_model_path}. '
        f'Run python tapt_training/pretrain_roberta.py first.'
    )

tokenizer = load_tokenizer(model_path=tapt_model_path, model_str=exp_configs.model_class)
data_util = DatasetUtil(dataset_name=dataset_name, max_length=exp_configs.max_seq_length, tokenizer=tokenizer)
dataset = data_util.get_dataset(dataset_name, split=None)
logger.info(f'Dataset {dataset_name} loaded. Starting pre-processing...')
train_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][0]])
val_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][2]])
num_id_labels = len(data_util.dataset_to_labels[dataset_name])

model = load_model(model_path=tapt_model_path, num_labels=num_id_labels)

trainer = get_trainer_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, model=model, tokenizer=tokenizer,
                                 num_epochs=exp_configs.num_train_epochs, batch_size=exp_configs.batch_size,
                                 output_dir=f"{exp_configs.temp_output_dir}/{trained_model_path}",
                                 seed=exp_configs.seed)

logger.info('\n\n\nTesting TAPT model on validation set...')
result = trainer.evaluate()
logger.info(result)
logger.info('Done. Starting supervised fine-tuning...\n\n\n')

if not debug_mode:
    result = trainer.train()
    print_summary(result)
    print('Training complete! Saving model...')

model.save_pretrained(savedir)
data_util.tokenizer.save_pretrained(savedir)
training_method = 'CE from TAPT checkpoint'
if debug_mode:
    training_method = f'{training_method} (debug mode: training skipped)'
print_model_save_details(savedir, dataset_name, training_method)
