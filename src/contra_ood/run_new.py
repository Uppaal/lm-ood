import os
import sys
import torch
import warnings
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig, GPT2Config, T5Config, AutoConfig

sys.path.append(os.getcwd() + '/..')
from evaluation import evaluate_ood
from utils_contra import set_seed, collate_fn
from setup import Map, logger, paths, run_configs
from model import RobertaForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification

# from model import prepare_ood ########### NEW

if run_configs.machine == 'local':
    from src.utils.dataset_utils import DatasetUtil
    from src.utils.model_utils import load_tokenizer
else:
    from utils.dataset_utils import DatasetUtil
    from utils.model_utils import load_tokenizer

warnings.filterwarnings("ignore")

task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20newsgroups': 20,
    # 'trec': 6,
    'news-category-modified': 17,
    'clinc150': 150,
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20newsgroups': 'mnli',
    # 'trec': 'mnli',
    'clinc150': 'mnli',
    'news-category-modified': 'mnli'
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):

    def detect_ood():
        # model = prepare_ood(model=model, is_train=True, dataloader=train_dataloader) ######### NEW
        # model = prepare_ood(model=model, is_train=False, dataloader=dev_dataloader) ######### NEW
        model.prepare_ood(train_dataloader)
        model.prepare_ood_val(dev_dataloader)
        logger.info(f'On train data, Dispersion={model.dispersion}; Compactness={model.compactness}')

        for tag, ood_features in benchmarks:
            ood_dataloader = torch.utils.data.DataLoader(ood_features, batch_size=args.batch_size, collate_fn=collate_fn)
            results = evaluate_ood(args, model, test_dataloader, ood_dataloader, tag=tag)
            logger.info('')
            for k, v in results.items():
                logger.info(f'{k}: {v}')
            torch.cuda.empty_cache()


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    if args.do_train:
        total_steps = int(len(train_dataloader) * args.num_train_epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)

        no_decay = ["LayerNorm.weight", "bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        if args.linear_probe:
            logger.info(f"Freezing weights...")
            for name, param in model.named_parameters():
                if name.startswith("roberta"):  # choose whatever you like here
                    param.requires_grad = False

        num_steps, best_test_acc = 0, 0.0
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f'Epoch: {epoch+1}')
            model.zero_grad()

            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                model.train()
                batch = {key: value.to(args.device) for key, value in batch.items()}
                outputs = model(**batch)
                loss, cos_loss = outputs[0], outputs[1]
                loss.backward()
                num_steps += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                pbar.set_postfix(loss=loss.item(), cos_loss=cos_loss.item())

            logger.info(f'Loss: {loss.item()} Cos Loss: {cos_loss.item()}')

            # Evaluation to study effect across epochs
            if args.epoch_wise_eval:
                results = evaluate(args, model, dev_dataloader, tag="test")
                logger.info(results)
                detect_ood(model, train_dataloader, dev_dataloader)
                model.save_pretrained(f'{args.savedir}_epoch-{epoch}')
                logger.info('\nSaving model...')

        if not args.epoch_wise_eval:
            logger.info('Training complete. Saving model...')
            model.save_pretrained(args.savedir)
            logger.info('Model saved.')

    if not args.epoch_wise_eval:
        results = evaluate(args, model, dev_dataloader, tag="dev")
        logger.info(results)
        detect_ood(model, train_dataloader, dev_dataloader)


def evaluate(args, model, dataloader, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()

        # Adding from my codebase
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
        acc = accuracy_score(labels, preds)
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        result['acc'] = acc

        return result

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["labels"] = None # To ensure loss isn't calculated in model.forward()
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results


def main(args):
    if run_configs.set_seed:
        set_seed(args)

    num_labels = task_to_labels[args.task_name]
    tokenizer = load_tokenizer(model_str=args.model_class)

    # Load model config
    if args.model_class == 'roberta':
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.layer_representation_for_ood = args.layer_representation_for_ood
    elif args.model_class == 'gpt2':
        config = GPT2Config.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    elif args.model_class == 't5':
        config = T5Config.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    config.gradient_checkpointing = True
    config.contrastive_weight = args.contrastive_weight
    config.cross_entropy_weight = args.cross_entropy_weight
    config.tau = args.tau
    config.report_all_metrics = args.report_all_metrics

    # Load model
    if args.model_class == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_class == 'gpt2':
        model = GPT2ForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        model.config.pad_token_id = model.config.eos_token_id
        model.config.use_cache = False
    elif args.model_class == 't5':
        model = T5ForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        # model.config.pad_token_id = model.config.eos_token_id
        # model.config.use_cache = False
    model.to(run_configs.device)
    logger.info(f'Loaded model. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


    datasets = args.ood_datasets
    benchmarks = ()

    def process_dataset_split(dataset_split):
        if args.samples_to_keep is not None:
            dataset_split = dataset_split.select(range(args.samples_to_keep))
        dataset_split = data_util.get_tensors_for_finetuning(dataset_split, format='torch')
        dataset_split = dataset_split.rename_column('label', 'labels')
        dataset_split = [x for x in dataset_split]

        processed_dataset_split = []
        for datum in dataset_split:
            if datum['labels'].numpy() == -1:
                datum['labels'] = torch.tensor(0, dtype=torch.int64)  # Following convention of original paper
            processed_dataset_split.append(datum)

        return processed_dataset_split

    for dataset_name in datasets:
        data_util = DatasetUtil(dataset_name=dataset_name, max_length=args.max_seq_length, tokenizer=tokenizer)
        dataset = data_util.get_dataset(dataset_name, split=None)

        if dataset_name == args.task_name:
            train_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][0]])
            test_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
            val_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][2]])

            # Since NC-Modified is a pure semantic shift dataset, the OOD benchmark comes from the same dataset
            if dataset_name == 'news-category-modified' or dataset_name == 'clinc150':
                test_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][3]])
                ood_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
                benchmarks = (('ood_' + dataset_name, ood_dataset),) + benchmarks

        else:
            ood_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
            benchmarks = (('ood_' + dataset_name, ood_dataset),) + benchmarks

    tokenizer.save_pretrained(args.savedir)
    train(args, model, train_dataset, val_dataset, test_dataset, benchmarks)
    logger.info('Done.')


if __name__ == "__main__":
    args = Map()

    args.task_name = '20newsgroups'  # '20newsgroups' 'imdb' 'news-category-modified'
    # args.ood_datasets = ['sst2', '20newsgroups', 'rte', 'mnli', 'imdb', 'multi30k', 'news-category-modified', 'clinc150']
    args.ood_datasets = ['rte'] + [args.task_name]
    args.model_class = 'roberta'  # 'roberta' 't5' 'gpt2'

    args.report_all_metrics = False
    args.epoch_wise_eval = False

    # RoBERTa configs
    args.layer_representation_for_ood = 'classifier_input'  # 'penultimate_layer' 'classifier_input'

    args.max_seq_length = 256
    args.do_train = True
    args.linear_probe = False
    args.tau = 0.1                   # Originally 0.3 - Temperature for SupCon
    args.contrastive_weight = 0.0    # Originally 2.0
    args.cross_entropy_weight = 1.0  # Originally 1.0

    args.model_name_or_path = paths.MODEL_DIR + f'/pretrained_models/{args.model_class}_base'
    # args.model_name_or_path = paths.MODEL_DIR + f'/pretrained_models/roberta_base_tapt-{args.task_name}/'

    args.savedir = paths.MODEL_DIR + f'/{args.model_class}_base_{args.task_name}_ce'
    # args.savedir = paths.MODEL_DIR + f'/{args.model_class}_base_{args.task_name}_tapt'
    # args.savedir = paths.MODEL_DIR + f'/{args.model_class}_base_{args.task_name}_contrastive_tau{args.tau}_alpha{args.contrastive_weight}'


    # Training Args
    args.samples_to_keep = 5 if run_configs.debug_mode else None
    args.batch_size = 4 if not run_configs.debug_mode else 2 # default 32
    args.learning_rate = 1e-5
    args.adam_epsilon = 1e-8  # Matching to HF default. Codebase original - 1e-6
    args.warmup_ratio = 0.0   # Matching to HF default. Codebase original -  0.06
    args.weight_decay = 0.01
    args.num_train_epochs = 10 if not run_configs.debug_mode else 1 # default 10
    args.seed = 42
    args.n_gpu = torch.cuda.device_count()
    args.device = run_configs.device

    if True:
        main(args)
        logger.info(f'Running with args {vars(args)}')
    else:
        args.task_name = '20newsgroups'
        args.ood_datasets = ['rte'] + [args.task_name]
        args.tau = 0.1
        args.contrastive_weight = 0.0

        parent_model_path = paths.MODEL_DIR + f'/finetuned_models/ce/id_20ng/roberta_base_20newsgroups_ce_epoch-'
        # parent_model_path = paths.MODEL_DIR + f'/finetuned_models/tapt/id_20ng/roberta_base_20newsgroups_tapt_epoch-'
        #parent_model_path = paths.MODEL_DIR + f'/finetuned_models/supcon/id_20ng/epoch_sweep/roberta_base_20newsgroups_contrastive_tau0.1_alpha2.0_epoch-'
        #parent_model_path = paths.MODEL_DIR + f'/finetuned_models/ce/id_imdb/roberta_base_imdb_contrastive_tau0.1_alpha0.0_epoch-'
        # parent_model_path = paths.MODEL_DIR + f'/finetuned_models/tapt/id_imdb/roberta_base_imdb_tapt_epoch-'
        #parent_model_path = paths.MODEL_DIR_local + f'/imdb_supcon/roberta_base_imdb_contrastive_tau0.7_alpha2.0_epoch-'
        # parent_model_path = paths.MODEL_DIR_local + f'/nc_ce/roberta_base_news-category-modified_ce_epoch-'
        #parent_model_path = paths.MODEL_DIR_local + f'/nc_tapt/roberta_base_news-category-modified_tapt_epoch-'
        #parent_model_path = paths.MODEL_DIR_local + f'/nc_supcon/roberta_base_news-category-modified_contrastive_tau0.1_alpha2.0_epoch-'


        for epoch in range(10):
            logger.info(f'\nEpoch {epoch}\n')
            args.model_name_or_path = f'{parent_model_path}{epoch}'
            main(args)
            logger.info(f'Running with args {vars(args)}')
            logger.info('\n\n')
            torch.cuda.empty_cache()
