from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=3,
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

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

