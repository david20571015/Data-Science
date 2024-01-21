# Crowd Estimation

## Environment

### Create environment

```bash
conda create --name hw4 python=3.10 -y
conda activate hw4

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install torchmetrics tqdm numpy pandas pillow scipy -c conda-forge -y
```

or

```bash
conda env create --file environment.yml
conda activate hw4
```

## Usage

### Data Location

- `data/`: The data folder.
  - `test/`: The testing data.
    - `*.jpg`: The testing images.
- `data_process/`: The data folder for data preprocess (given by TA).
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*.txt`: The training labels.

### Data Preprocess

```bash
# Resize labels to fit the image size and compute the size of bbox.
python data_process.py --origin-dir data_process --data-dir data_processed
```

After running the above commands, the data folder should look like this:

- `data/`: The data folder.
  - `test/`: The testing data.
    - `*.jpg`: The testing images.
- `data_process/`: The data folder for data preprocess.
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*.txt`: The training labels.
- `data_processed/`: The data folder for preprocessed data.
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*.npy`: The training labels.

`data_process` is able to be deleted after running the above commands.

### Train

train with default parameters

```bash
python train.py
```

or

```bash
python train.py [-h] [--data-dir DATA_DIR] [--epoch EPOCH] [--lr LR] [--weight-decay WEIGHT_DECAY] [--sigma SIGMA]
```

options:

- `-h, --help`            show this help message and exit
- `--data-dir DATA_DIR`   path to training data image directory which contains images(\*.jpg) and points(\*.npy)
                          (default: data_processed/train)
- `--epoch EPOCH`         epoch (default: 1000)
- `--lr LR`               learning rate (default: 1e-05)
- `--weight-decay WEIGHT_DECAY`
                          weight decay (default: 0.0001)
- `--sigma SIGMA`         sigma for gaussian kernel (default: 8.0)

### Predict

predict with default parameters

```bash
python predict.py
```

or

```bash
python predict.py [-h] [--data DATA] [--weights WEIGHTS]
```

options:

- `-h, --help`         show this help message and exit
- `--data DATA`        path to dataset which contains images(\*.jpg) (default: data/test)
- `--weights WEIGHTS`  path to weights (default: best_model.pth)

### Predict Result

- `pred.csv`: A csv file with two columns: `ID` and `COUNT`.

## Method

[Bayesian Loss for Crowd Count Estimation with Point Supervision](https://arxiv.org/pdf/1908.03684.pdf)
