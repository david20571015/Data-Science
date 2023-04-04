import argparse
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.dataset import TrainDataset
from src.model import ConvNet
from src.sampler import CategoriesSampler
from src.utils import euclidean_distance_sqaured


def main():
    train_dataset = TrainDataset('data/train.pkl')
    train_sampler = CategoriesSampler(train_dataset.labels,
                                      n_classes=args.train_n_class,
                                      n_batch=400,
                                      n_way=args.train_n_way,
                                      n_shot=args.n_shot,
                                      n_query=args.n_query)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True)

    valid_dataset = TrainDataset('data/validation.pkl')
    valid_sampler = CategoriesSampler(valid_dataset.labels,
                                      n_classes=args.valid_n_class,
                                      n_batch=100,
                                      n_way=args.valid_n_way,
                                      n_shot=args.n_shot,
                                      n_query=args.n_query)
    valid_loader = DataLoader(valid_dataset,
                              batch_sampler=valid_sampler,
                              num_workers=4,
                              pin_memory=True)

    model = ConvNet(emb_size=args.emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
                                                gamma=0.5)

    best_valid_acc = 0
    best_state_dict = None

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')

        train_loss = MeanMetric().to(DEVICE)
        train_acc = Accuracy(task='multiclass',
                             num_classes=args.train_n_way).to(DEVICE)
        model.train()
        with tqdm(train_loader, desc=f'train', dynamic_ncols=True) as pbar:
            for data, label in pbar:
                # data.shape: (n_way, n_shot + n_query, channel, width, height)
                data = data.to(DEVICE)
                support, query = data[:, :args.n_shot], data[:, args.n_shot:]

                support = support.reshape(-1, *support.shape[-3:])
                proto = model(support)
                proto = proto.reshape(args.train_n_way, args.n_shot, -1)
                proto = proto.mean(dim=1)
                # proto.shape: (n_way, -1)

                query = query.reshape(-1, *query.shape[-3:])
                output = model(query)
                output = output.reshape(args.train_n_way * args.n_query, -1)

                # logits.shape: (n_way * n_query, n_way)
                logits = -euclidean_distance_sqaured(output, proto)

                labels = torch.arange(
                    args.train_n_way,
                    dtype=torch.long,
                    device=DEVICE,
                ).repeat(args.n_query, 1).t().reshape(-1)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.update(loss)
                train_acc.update(logits.argmax(dim=-1), labels)

                pbar.set_postfix_str(
                    f'loss: {train_loss.compute():.4f}, acc: {train_acc.compute():.4f}'
                )

        scheduler.step()

        valid_loss = MeanMetric().to(DEVICE)
        valid_acc = Accuracy(task='multiclass',
                             num_classes=args.valid_n_way).to(DEVICE)

        model.eval()
        with tqdm(valid_loader, desc=f'valid',
                  dynamic_ncols=True) as pbar, torch.no_grad():
            for data, _ in pbar:
                # data.shape: (n_way, n_shot + n_query, channel, width, height)
                data = data.to(DEVICE)
                support, query = data[:, :args.n_shot], data[:, args.n_shot:]

                support = support.reshape(-1, *support.shape[-3:])
                proto = model(support)
                proto = proto.reshape(args.valid_n_way, args.n_shot, -1)
                proto = proto.mean(dim=1)
                # proto.shape: (n_way, -1)

                query = query.reshape(-1, *query.shape[-3:])
                output = model(query)
                output = output.reshape(args.valid_n_way * args.n_query, -1)

                # logits.shape: (n_way * n_query, n_way)
                logits = -euclidean_distance_sqaured(output, proto)

                labels = torch.arange(
                    args.valid_n_way,
                    dtype=torch.long,
                    device=DEVICE,
                ).repeat(args.n_query, 1).t().reshape(-1)

                loss = F.cross_entropy(logits, labels)

                valid_loss.update(loss)
                valid_acc.update(logits.argmax(dim=-1), labels)

                pbar.set_postfix_str(
                    f'loss: {valid_loss.compute():.4f}, acc: {valid_acc.compute():.4f}'
                )

        if valid_acc.compute() > best_valid_acc:
            best_valid_acc = valid_acc.compute()
            best_state_dict = copy.deepcopy(model.state_dict())

    print(f'Best valid acc: {best_valid_acc:.4f}')
    torch.save(best_state_dict, 'model.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--train-n-class', type=int, default=64)
    parser.add_argument('--valid-n-class', type=int, default=16)
    parser.add_argument('--train-n-way', type=int, default=20)
    parser.add_argument('--valid-n-way', type=int, default=5)
    parser.add_argument('--n-shot', type=int, default=5)
    parser.add_argument('--n-query', type=int, default=5)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
