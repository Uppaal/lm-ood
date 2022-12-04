import os
import sys
sys.path.append(os.getcwd() + '/..')

from setup import run_configs, paths, logger, print_summary
from train_utils import get_trainer_finetuning
debug_mode = run_configs.debug_mode

# if run_configs.machine == 'local':
#     from src.utils.model_utils import load_model
#     from src.utils.dataset_utils import DatasetUtil
# else:
from utils_ood.model_utils import load_model
from utils_ood.dataset_utils import DatasetUtil


sys.path.extend([(paths.root_path + source_path) for source_path in os.listdir(paths.root_path) if not (paths.root_path + source_path) in sys.path])
debug_mode = run_configs.debug_mode


''' Declare params '''
# dataset_name = 'sst2'
dataset_name = '20newsgroups'
# dataset_name = '20newsgroups-open-set-classification'
# dataset_name = 'clinc150'
# dataset_name = 'news-category-modified'

original_model_name = f'pretrained_models/roberta_base'
trained_model_name = f'/roberta_base_{dataset_name}'
# original_model_name = f'pretrained_models/roberta_base_tapt-{dataset_name}'
# trained_model_name = f'/roberta_base_tapt-{dataset_name}_finetuned'

savedir = paths.MODEL_DIR + trained_model_name
data_util = DatasetUtil(dataset_name=dataset_name, max_length=run_configs.max_seq_len)

''' Load and process dataset '''
dataset = data_util.get_dataset(dataset_name, split=None)
logger.info(f'Dataset {dataset_name} loaded. Starting pre-processing...')
train_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][0]])
val_dataset = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][2]])

# num_id_labels = len(data_util.id_classes)
num_id_labels = len(data_util.dataset_to_labels[dataset_name])

''' Load model '''
model, tokenizer = load_model(model_name=original_model_name, num_labels=num_id_labels)

''' Training '''

trainer = get_trainer_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, model=model, tokenizer=tokenizer,
                                 num_epochs=3, batch_size=4, output_dir=f"{paths.temp_output_dir}/{trained_model_name}")

logger.info('\n\n\nTesting pretrained model on test set...')
# To get list of all metrics: datasets.list_metrics()
# Some important metrics: 'accuracy', 'exact_match', 'f1', 'mahalanobis', 'precision', 'recall', 'mse'
result = trainer.evaluate()
logger.info(result)
logger.info('Done. Starting training...\n\n\n')

if not debug_mode:
    result = trainer.train()
    print_summary(result)
    print('Training complete! Saving model...')

model.save_pretrained(savedir)
data_util.tokenizer.save_pretrained(savedir)
print('Model saved.')

# Eval on test set automatically happens at the end of training.
