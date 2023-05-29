import json
from typing import cast

from datasets import DatasetDict
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

MODEL_NAME = './model'

dataset = cast(
    DatasetDict,
    load_dataset('json',
                 data_files='data/test.json',
                 keep_in_memory=True,
                 num_proc=5))
dataset = dataset['train']


def preprocess_func(data):
    prefix = 'summarize: '
    inputs = [prefix + (body or '') for body in data['body']]
    return {'inputs': inputs}


dataset = dataset.map(preprocess_func,
                      batched=True,
                      remove_columns=['body'],
                      num_proc=5)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          use_fast=True,
                                          model_max_length=512,
                                          local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,
                                              local_files_only=True)

summarizer = pipeline('summarization',
                      model=model,
                      tokenizer=tokenizer,
                      device='cuda:0')

with open('311511038.json', 'a', encoding='utf-8') as f:
    for output in tqdm(summarizer(KeyDataset(dataset, 'inputs'),
                                  batch_size=32,
                                  truncation=True),
                       total=len(dataset)):
        result = output[0]['summary_text']
        print(json.dumps({'title': result}), file=f)

    print(file=f) # add a newline at the end
