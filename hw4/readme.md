# Data Science HW4

## Environment

### Create environment

```bash
conda create --name hw3 python=3.10 -y
conda activate hw4

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install numpy tqdm pandas torchmetrics -c conda-forge -y
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
# resize labels to fit the image size
# take about 10 minutes on 1 GeForce GTX 1080
python data_process.py --origin-dir data_process --data-dir data_processed

# create density map and save to data/train
# take about 10 minutes on 1 GeForce GTX 1080
# you can remove the `--dynamic` option to use static sigma which leads to poor performance but faster speed
python create_dm.py --input-dir data_processed/train --output-dir data/train --dynamic

# create density map and save to data/train
# take about 50 minutes on 1 GeForce GTX 1080
python create_dm.py --input-dir data_processed/train --output-dir data/train

cp data_processed/train/* data/train
```

After running the above commands, the data folder should look like this:

- `data/`: The data folder.
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*_dm.jpg`: The training density maps.
    - `*.npy`: The training labels.
  - `test/`: The testing data.
    - `*.jpg`: The testing images.
- `data_process/`: The data folder for data preprocess.
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*.txt`: The training labels.
- `data_processed/`: The data folder for data preprocess.
  - `train/`: The training data.
    - `*.jpg`: The training images.
    - `*.npy`: The training labels.

Both `data_process` and `data_processed` are able to be deleted after running the above commands.

### Train

train with default parameters

```bash
python train.py
```

or

```bash
python train.py [-h] [--train-data TRAIN_DATA] [--valid-data VALID_DATA] [--dims DIMS [DIMS ...]] [--emb-dim EMB_DIM]
                [--epochs EPOCHS] [--train-n-class TRAIN_N_CLASS] [--valid-n-class VALID_N_CLASS] [--train-n-way TRAIN_N_WAY]
                [--valid-n-way VALID_N_WAY] [--n-shot N_SHOT] [--n-query N_QUERY]
```

options:

- `-h, --help`: show this help message and exit
- `--train-data TRAIN_DATA`: path to train pkl data. default: data/train.pkl
- `--valid-data VALID_DATA`: path to validation pkl data. default: data/validation.pkl
- `--dims DIMS [DIMS ...]`: channels of hidden conv layers. default: [64, 64, 64]
- `--emb-dim EMB_DIM`: embedding dimension (output dim of the model). default: 64
- `--epochs EPOCHS`: default: 300
- `--train-n-class TRAIN_N_CLASS`: number of total classes in the training dataset. default: 64
- `--valid-n-class VALID_N_CLASS`: number of total classes in the valid dataset. default: 16
- `--train-n-way TRAIN_N_WAY`: number of classes in a training episode. default: 20
- `--valid-n-way VALID_N_WAY`: number of classes in a valid episode. default: 5
- `--n-shot N_SHOT`: number of support examples per class. default: 5
- `--n-query N_QUERY`: number of query examples per class. default: 5

### Predict

predict with default parameters

```bash
python test.py
```

or

```bash
python test.py [-h] [--test-data TEST_DATA] [--dims DIMS [DIMS ...]] [--emb-dim EMB_DIM] [--weights WEIGHTS]
```

options:

- `-h, --help`: show this help message and exit
- `--test-data TEST_DATA`: path to test pkl data. default: data/test.pkl
- `--dims DIMS [DIMS ...]`: channels of hidden conv layers. default: [64, 64, 64]
- `--emb-dim EMB_DIM`: embedding dimension (output dim of the model). default: 64
- `--weights WEIGHTS`: path to model weights. default: model.pth

### Predict Result

- `pred.csv`: A csv file with two columns: `id` and `category`.

## Method

[Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
