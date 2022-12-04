import os
import sys
sys.path.append(os.getcwd() + '/..')
os.environ["WANDB_DISABLED"] = "true"

from setup import run_configs, paths, logger, print_summary
debug_mode = run_configs.debug_mode

from train_utils import get_trainer_pretraining
if run_configs.machine == 'local':
    from src.utils.dataset_utils import DatasetUtil
    from src.utils.model_utils import load_model
else:
    from utils.dataset_utils import DatasetUtil
    from utils.model_utils import load_model


# def tokenize_function(examples):
#     return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=run_configs.max_seq_len)

''' Declare params '''
# dataset_name = 'imdb'
# dataset_name = '20newsgroups'
# dataset_name = '20newsgroups-open-set'
# dataset_name = 'clinc150'
dataset_name = 'news-category-modified'

original_model_name = 'pretrained_models/roberta_base'
trained_model_name = f'/roberta_base_tapt-{dataset_name}'

''' Load model '''
model, tokenizer = load_model(model_name=original_model_name, load_only_body=True)
savedir = paths.MODEL_DIR + trained_model_name
data_util = DatasetUtil(dataset_name=dataset_name, max_length=run_configs.max_seq_len)

# import numpy as np
# def tokenize_function_with_masking(examples, mlm_probability=0.15):
#     output = tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=run_configs.max_seq_len)
#     mask_token_id = tokenizer.vocab['<mask>']
#     input_ids = np.array(output['input_ids'])
#
#     mask_ids = np.random.choice([0, 1], size=input_ids.shape, p=[.85, mlm_probability])
#     mask_ids[:][0] = 0
#     mask_ids[:][-1] = 0
#     input_ids = np.where(mask_ids == 1, input_ids, mask_token_id)[:5][:5]
#
#     output['input_ids'] = list(input_ids)
#     return output
#
# dataset = data_util.get_dataset(dataset_name, split=None)
# from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# train_split = dataset['oos_train']
# train_dataset = train_split.map(tokenize_function_with_masking, batched=True, remove_columns=['sentence', 'label', 'label_text'])
# data_collator.numpy_mask_tokens(train_dataset)

''' Load and process dataset '''
dataset = data_util.get_dataset(dataset_name, split=None)
logger.info(f'Dataset {dataset_name} loaded. Starting pre-processing...')
train_split = dataset[data_util.dataset_to_split[dataset_name][0]]
train_dataset = train_split.map(data_util.preprocess_function, batched=True,
    remove_columns=[x for x in data_util.dataset_to_keys[data_util.dataset_name] if x is not None])

# Ref: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb#scrollTo=iaAJy5Hu3l_B
# def group_texts(examples, block_size=512):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the
#     # model supported it instead of this drop, you can customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result
#
# train_dataset = train_dataset.map(
#     group_texts,
#     batched=True,
#     batch_size=1000,
#     num_proc=4,)

''' Training '''

trainer = get_trainer_pretraining(dataset=train_dataset, model=model, tokenizer=tokenizer, batch_size=16,
                                  num_epochs=3, output_dir=f"{paths.temp_output_dir}/pretrain_results_tapt-{dataset_name}")
# TAPT paper does 100 epochs, but OOD papers finetune for only 3 epochs.

logger.info('Starting training...\n\n\n')
result = trainer.train()
print('Training complete! Saving model...')
model.save_pretrained(savedir)
data_util.tokenizer.save_pretrained(savedir)
print('Model saved.')

print(result)
print_summary(result)