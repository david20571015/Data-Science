from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import torch
from torch.nn.utils import prune


def prune_model(model: torch.nn.Module, prune_ratio=0.2):
    prune_params = []

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune_params.extend([(module, 'weight')])
        if isinstance(module, torch.nn.Linear):
            prune_params.extend([(module, 'weight'), (module, 'bias')])

    prune.global_unstructured(prune_params,
                              pruning_method=prune.L1Unstructured,
                              amount=prune_ratio)


def remove_prune(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')


def mixup(data, target):
    lam = torch.rand(1, device=data.device)
    index = torch.randperm(data.size(0), device=data.device)

    mixed_data = torch.lerp(data, data[index], lam)
    mixed_target = torch.lerp(target, target[index], lam)

    return mixed_data, mixed_target


MonitorType = Literal['train_acc', 'train_loss', 'test_acc', 'test_loss']
CriteriaType = Union[float, torch.Tensor]


class Recorder:

    def __init__(
        self,
        logdir: str,
        max_to_keep: Optional[int] = None,
        save_best=True,
        moniter: MonitorType = 'test_acc',
    ) -> None:
        self.logdir = 'logs' / Path(logdir)
        self.logdir.mkdir(exist_ok=True, parents=True)

        self.log_filename = self.logdir / 'log.txt'

        self.max_to_keep = max_to_keep

        self.save_best = save_best
        self.moniter = moniter
        self.mode = 'max' if moniter.endswith('acc') else 'min'

        self.best_criterias = float('-inf' if self.mode == 'max' else 'inf')
        self.best_filename = self.logdir / 'best.pth'

    def _is_better(self, criteria: CriteriaType):
        if self.mode == 'max':
            return criteria > self.best_criterias
        return criteria < self.best_criterias

    def _save_checkpoint(self, state_dict: Dict[str, Any], epoch: int):
        if self.max_to_keep:
            (self.logdir / f'epoch_{epoch-self.max_to_keep}.pth').unlink(True)
        torch.save(state_dict, self.logdir / f'epoch_{epoch}.pth')

    def _save_best_checkpoint(self, state_dict: Dict[str, Any],
                              criteria: CriteriaType):
        self.best_criterias = criteria

        self.best_filename.unlink(missing_ok=True)
        self.best_filename = self.logdir / f'best.pth'

        torch.save(state_dict, self.best_filename)

        with open(self.log_filename, 'a', encoding='utf-8') as f:
            print(
                f'Best model saved at {self.best_filename}, {self.moniter}: {criteria}',
                file=f)

    def update(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        criteria: CriteriaType,
        log_dict: Dict[str, CriteriaType] = {},
    ):
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch}, {self.moniter}: {criteria:.4f} - |')
            for key, value in log_dict.items():
                f.write(f' {key}: {value:.4f} |')
            f.write('\n')

        self._save_checkpoint(state_dict, epoch)

        if self.save_best and self._is_better(criteria):
            self._save_best_checkpoint(state_dict, criteria)

    def log(self, filename: str, content: Any):
        with open(self.logdir / filename, 'a', encoding='utf-8') as f:
            print(content, file=f)


class EarlyStopper:

    def __init__(
        self,
        patience=1,
        min_delta=0.0,
        moniter: MonitorType = 'test_acc',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.moniter = moniter

        self.counter = 0
        self.mode = 'max' if moniter.endswith('acc') else 'min'

        self.best_epoch = 0
        self.best_criteria = float('-inf' if self.mode == 'max' else 'inf')

    def update(self, epoch: int, criteria: CriteriaType):
        if self.mode == 'max':
            if criteria > self.best_criteria:
                self.best_epoch = epoch
                self.best_criteria = criteria
                self.counter = 0
            elif criteria < self.best_criteria - self.min_delta:
                self.counter += 1
        else:
            if criteria < self.best_criteria:
                self.best_epoch = epoch
                self.best_criteria = criteria
                self.counter = 0
            elif criteria > self.best_criteria + self.min_delta:
                self.counter += 1

    @property
    def stop(self):
        return self.counter >= self.patience

    def best_state(self):
        return self.best_epoch, self.moniter, self.best_criteria
