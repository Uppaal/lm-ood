from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics_(pred):
    # pred = model(input_ids=train_dataset[:5]['input_ids'].to(run_configs.device), attention_mask=train_dataset[:5]['attention_mask'].to(run_configs.device))
    # from transformers import RobertaConfig, RobertaForSequenceClassification
    # model_config = RobertaConfig.from_pretrained(model_path, num_labels=num_id_labels, output_hidden_states=True)
    # model = RobertaForSequenceClassification.from_pretrained(model_path, config=model_config)

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc.tolist(),
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }

def compute_metrics(pred):
    # Ref source code: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L100
    # pred.label_ids or pred[0] - (N, )
    # pred.predictions or pred[1] - tuple
    # pred.predictions[0] - (N, C)
    # pred.predictions[1] - tuple of 13 arrays of shape (N, B, H)

    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1) #  Added [0] for making consistent with new HF update
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc.tolist(),
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }


def get_trainer_finetuning(train_dataset, val_dataset, model, tokenizer, num_epochs, batch_size, output_dir):

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,  # Originally 16
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=3,    # Saves n most recent models
        # save_steps=5000      # Save every n steps of training
        # load_best_model_at_end= True # Whether or not to load the best model found during training at the end of training.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


def get_trainer_pretraining(dataset, model, tokenizer, num_epochs, batch_size, output_dir):
    # Ref: https://towardsdatascience.com/transformers-retraining-roberta-base-using-the-roberta-mlm-procedure-7422160d5764
    # Code of paper: https://github.com/allenai/dont-stop-pretraining/blob/266269faca8645482eef2e710d916607ea2c71d2/scripts/run_language_modeling.py#L69

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # data_collator.numpy_mask_tokens(dataset)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        learning_rate=0.0001,
        per_device_train_batch_size=batch_size, # Paper uses overall batch size of 256 when < 5k DPs
        save_steps=5000,
        save_total_limit=2,
        seed=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    return trainer


def _train(model, tokenizer, optimizer, criterion, device, train_loader, num_epochs, output_dir):
    from tqdm import tqdm, trange
    losses = []
    train_iterator = trange(int(num_epochs), desc='Epoch')
    for _ in train_iterator:
        tr_loss = 0
        step = None
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
            labels = batch[2].to(device)

            optimizer.zero_grad()

            out = model(**inputs)[1].double().to(device)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
        losses.append(tr_loss/(step+1))
        print('train loss: {}'.format(tr_loss/(step+1)))

    # save model and tokenizer
    print('Saving model and tokenizer')

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
