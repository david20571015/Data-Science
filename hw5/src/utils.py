from functools import partial
from typing import cast

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk


def prepare_train_dataset(
    tokenizer,
    data_files='data/train.json',
    cache_dir='cache/train',
):
    try:
        dataset = cast(Dataset, load_from_disk(cache_dir))

        features = ['input_ids', 'attention_mask', 'labels']

        for feature in features:
            if feature not in dataset.features:
                raise ValueError(
                    f'Feature {feature} not found in cached dataset')

        print('Load cached dataset')
        return dataset

    except (ValueError, FileNotFoundError) as err:
        print(f'Unable to load cached dataset because: {err}')
        print('Creating dataset from scratch')

    def filter_none(data, column_names):
        is_none = [map(bool, data[col]) for col in column_names]
        return list(map(all, zip(*is_none)))

    def preprocess_func(data):
        prefix = 'summarize: '
        inputs = [prefix + body for body in data['body']]

        model_inputs = tokenizer(inputs, truncation=True)

        labels = tokenizer(data['title'], truncation=True)

        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    dataset = cast(DatasetDict,
                   load_dataset('json', data_files=data_files, num_proc=5))
    dataset = dataset['train']

    dataset = dataset.filter(partial(filter_none,
                                     column_names=dataset.column_names),
                             batched=True,
                             num_proc=5)

    dataset = dataset.map(preprocess_func,
                          batched=True,
                          remove_columns=['title', 'body'],
                          num_proc=5)

    dataset.save_to_disk(cache_dir)

    return dataset
