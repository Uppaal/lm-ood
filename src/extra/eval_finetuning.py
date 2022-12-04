import os
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import RobertaConfig, RobertaForSequenceClassification, TrainingArguments, Trainer

from utils.dataset_utils import DatasetUtil
from src.setup import run_configs, paths, logger
sys.path.extend([(paths.root_path + source_path) for source_path in os.listdir(paths.root_path) if not (paths.root_path + source_path) in sys.path])
debug_mode = run_configs.debug_mode


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir=f"{paths.temp_output_dir}/finetune_results_",
    evaluation_strategy="epoch",
    save_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,  # Originally 16
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,    # Saves n most recent models
    # save_steps=5000      # Save every n steps of training
    # load_best_model_at_end= True # Whether or not to load the best model found during training at the end of training.
)

max_length = 512

def eval_on_data(dataset_name):

    data_util = DatasetUtil(dataset_name=dataset_name, lm_model='roberta_base', model_dir=paths.MODEL_DIR,
                            hf_cache_dir=None, max_length=max_length, debug_mode=debug_mode)


    dataset = data_util.get_tensors_end_to_end(split=data_util.dataset_to_split[dataset_name][-1])

    def eval_model(model_path):
        model_config = RobertaConfig.from_pretrained(model_path,
                                                     num_labels=len(set(dataset['label'].data.numpy())))  # Note: "max_position_embeddings": 514, unlike BERT (512); but embedding size is still 512: https://github.com/pytorch/fairseq/issues/1187
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=model_config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            tokenizer=data_util.tokenizer,
            compute_metrics=compute_metrics,
        )
        logger.info('\n\nStarting evaluation on test set. NOTE: If dealing with a semantic shift dataset (NewsCategory or DBPedia), train and test sets are the same.')
        logger.info(f'Test dataset: {dataset_name} Model: {model_path}')
        metrics = trainer.evaluate()
        logger.info(f'Metrics: {metrics}')
        logger.info('Done.\n\n')

    eval_model(paths.MODEL_DIR + f'/roberta_base')
    eval_model(paths.MODEL_DIR + f'/roberta_base_yelp_polarity')
    eval_model(paths.MODEL_DIR + f'/roberta_base_imdb')
    eval_model(paths.MODEL_DIR + f'/roberta_base_sst2')


eval_on_data(dataset_name='sst2') # All models (Base, IMDB, SST-2, Yelp) on SST-2
eval_on_data(dataset_name='imdb') # All models (Base, IMDB, SST-2, Yelp) on IMDB
eval_on_data(dataset_name='yelp_polarity') # All models (Base, IMDB, SST-2, Yelp) on Yelp
