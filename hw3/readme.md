# Data Science HW3

## Environment

### Create environment

```bash
conda create --name hw3 python=3.10 -y
conda activate hw3

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install tqdm pandas torchmetrics -c conda-forge -y
```

or

```bash
conda env create --file environment.yml
conda activate hw3
```

## Usage

### Data Location

- `data/`: The data folder.
  - `train.pkl`: The training data.
  - `validation.pkl`: The validation data.
  - `test.cpklsv`: The testing data.

### Train

```bash
python train.py
```

### Predict

```bash
python test.py --weights <weight_path>
```

or use pretraind weight

```bash
python test.py --weights model.pth
```

### Predict Result

- `pred.csv`: A csv file with two columns: `id` and `category`.

## Method

[Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
