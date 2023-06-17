import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch_geometric.nn as geom_nn

from src.datasets import GraphDataModule
from src.models import NodeLevelGNN


def main():
    datamodule = GraphDataModule()
    model = NodeLevelGNN(base_model=geom_nn.GCN,
                         model_kwargs={
                             'in_channels': -1,
                             'hidden_channels': 32,
                             'out_channels': 1,
                             'num_layers': 6,
                             'dropout': 0.1,
                             'act': 'relu',
                             'norm': 'instance_norm',
                             'jk': 'cat',
                         })

    trainer = pl.Trainer(max_epochs=10000,
                         benchmark=True,
                         precision='64-true',
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True,
                                             mode='max',
                                             monitor='val_auc'),
                         ],
                         logger=[
                             TensorBoardLogger(save_dir='logs',
                                               name='',
                                               default_hp_metric=False)
                         ],
                         default_root_dir=None)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
