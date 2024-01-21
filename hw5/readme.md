# Headline Generation

## Environment

### Create environment

```bash
conda env create --file environment.yml
conda activate hw5
```

## Usage

### Data Location

- `data/`: The data folder.
  - `train.json`: The training data.
  - `test.json`: The testing data.

### Train

```bash
python train.py
```

### Predict

```bash
python infer.py
```

### Predict Result

- `submission.json`: A json file with 13762 json lines.
  - `title`: Generated news title in string format.

## Method

In this homework, I followed the tutorial provided by [Hugging Face](https://huggingface.co/docs/transformers/tasks/summarization) and fine-tuned the [`T5-small`](https://huggingface.co/t5-small) model to generate the title.
