from argparse import ArgumentParser

import lightning as pl
import pandas as pd
import torch

from src.datasets import GraphDataModule
from src.models import NodeLevelGNN


def main():
    datamodule = GraphDataModule()
    model = NodeLevelGNN.load_from_checkpoint(args.weights).eval()

    trainer = pl.Trainer(precision='64-true')

    with torch.inference_mode():
        predictions, idx = trainer.predict(model, datamodule=datamodule)[0]

    df = pd.DataFrame({
        'node idx': idx.tolist(),
        'node anomaly score': predictions.flatten().tolist()
    }).set_index('node idx')

    df.to_csv('submission.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights',
                        '-w',
                        type=str,
                        required=True,
                        help='Path to the model weights.')
    args = parser.parse_args()

    main()
