# Data Science HW6

## Environment

### Create environment

```bash
conda env create --file environment.yml
conda activate hw6
```

## Usage

### Data Location

- `data/`: The data folder.
  - `train_mask.npy`: The mask of training data.
  - `train_sub-graph_tensor.pt`: The sub-graph tensor of training data.
  - `test_mask.npy`: The mask of testing data.
  - `test_sub-graph_tensor_noLabel.pt`: The sub-graph tensor of testing data.

### Train

```bash
python3 train.py
```

### Predict

```bash
python3 infer.py
```

### Predict Result

- `submission.csv`: The prediction result.
  - `node idx`: The node index.
  - `node anomaly score`: The anomaly score of the node.
