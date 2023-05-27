import os
from typing import cast

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from src.utils import prepare_train_dataset

MODEL_NAME = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          use_fast=False,
                                          model_max_length=512)

dataset = prepare_train_dataset(tokenizer, cache_dir='cache/train')
dataset.set_format(type='torch', columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.2)
train_dataset, test_dataset = dataset['train'], dataset['test']

print(dataset)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

metric_rouge = evaluate.load('rouge',
                             rouge_types=['rouge1', 'rouge2', 'rougeL'])
metric_bertscore = evaluate.load('bertscore')


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge = metric_rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    bertscore = metric_bertscore.compute(predictions=decoded_preds,
                                         references=decoded_labels,
                                         lang='en')
    rouge = cast(dict, rouge)
    bertscore = cast(dict, bertscore)
    bertscore.pop('hashcode')
    bertscore = {k: np.mean(v).item() for k, v in bertscore.items()}
    return rouge | bertscore


training_args = Seq2SeqTrainingArguments(
    output_dir="logs",
    save_strategy="steps",
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    weight_decay=1e-3,
    save_total_limit=3,
    num_train_epochs=150,
    predict_with_generate=True,
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    report_to=['tensorboard'],
    logging_strategy='steps',
    logging_steps=1000,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), None),
)

trainer.train(resume_from_checkpoint=True)

trainer.save_model('model')
