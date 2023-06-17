from typing import Any

import lightning as pl
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch_geometric.nn as geom_nn


class NodeLevelGNN(pl.LightningModule):

    def __init__(
        self,
        base_model: type[geom_nn.models.basic_gnn.BasicGNN],
        model_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = base_model(**model_kwargs)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def _compute_loss(self, y_pred, y_true):
        pos_weight = (y_true == 0.0).sum() / (y_true == 1.0).sum()
        return F.binary_cross_entropy_with_logits(y_pred,
                                                  y_true,
                                                  pos_weight=pos_weight)

    def _compute_auc(self, y_pred, y_true):
        y_pred_np = y_pred.detach().sigmoid().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        return roc_auc_score(y_true_np, y_pred_np)

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y, batch.edge_index
        train_mask = batch.train_mask

        x = self(x, edge_index)

        x, y = x[train_mask], y[train_mask]
        loss = self._compute_loss(x, y)
        auc = self._compute_auc(x, y)

        logs = {'train_loss': loss, 'train_auc': auc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y, batch.edge_index
        val_mask = batch.valid_mask

        x = self(x, edge_index)

        x, y = x[val_mask], y[val_mask]
        loss = self._compute_loss(x, y)
        auc = self._compute_auc(x, y)

        logs = {'val_loss': loss, 'val_auc': auc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        predict_mask = batch.test_mask

        x = self(x, edge_index)
        x = x[predict_mask].sigmoid()

        idx = torch.arange(batch.num_nodes,
                           dtype=torch.long,
                           device=self.device)[predict_mask]
        return x, idx
