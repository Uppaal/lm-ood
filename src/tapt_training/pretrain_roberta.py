import os
from src.utils.dataset_utils import DatasetUtil
from src.utils.trainer_utils import get_trainer_pretraining
from src.utils.model_utils import load_model, load_tokenizer
from src.utils.startup import exp_configs, logger, print_summary

os.environ["WANDB_DISABLED"] = "true"

dataset_name = exp_configs.task_name
original_model_path = exp_configs.model_name_or_path
savedir = exp_configs.MODEL_DIR + f'/pretrained_models/{exp_configs.model_class}_base_{dataset_name}_tapt'
debug_mode = exp_configs.debug_mode

tokenizer = load_tokenizer(model_str=exp_configs.model_class)
data_util = DatasetUtil(dataset_name=dataset_name, max_length=exp_configs.max_seq_length, tokenizer=tokenizer)

model = load_model(model_path=original_model_path, load_only_body=True)

dataset = data_util.get_dataset(dataset_name, split=None)
logger.info(f'Dataset {dataset_name} loaded. Starting pre-processing...')
train_split = dataset[data_util.dataset_to_split[dataset_name][0]]
train_dataset = train_split.map(data_util.preprocess_function, batched=True,
    remove_columns=[x for x in data_util.dataset_to_keys[data_util.dataset_name] if x is not None])

trainer = get_trainer_pretraining(dataset=train_dataset, model=model, tokenizer=tokenizer, batch_size=16,
                                  num_epochs=exp_configs.num_train_epochs,
                                  output_dir=f"{exp_configs.temp_output_dir}/pretrain_results_tapt-{dataset_name}")

logger.info('Starting tapt_training...\n\n\n')
result = trainer.train()
print('Training complete! Saving model...')
model.save_pretrained(savedir)
data_util.tokenizer.save_pretrained(savedir)
print('Model saved.')

print(result)
print_summary(result)