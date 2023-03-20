from datetime import datetime
import os

import torch
import torch.nn.functional as F
import torch.utils.data
from torchinfo import summary
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
import torchvision
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet50
import torchvision.transforms as T
from tqdm import tqdm

from utilis import mixup
from utilis import prune_model
from utilis import Recorder


def get_dataloaders(batch_size=256):
    transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, ))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=6,
                                              pin_memory=True)

    testset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size * 4,
                                             shuffle=False,
                                             num_workers=6,
                                             pin_memory=True)

    return trainloader, testloader


def compute_loss(output, target, teacher_output=None, temp=1.0, alpha=0.5):
    hard_loss = F.cross_entropy(output, target)
    if teacher_output is None:
        return hard_loss

    soft_loss = F.kl_div(torch.log_softmax(output / temp, dim=-1),
                         torch.log_softmax(teacher_output / temp, dim=-1),
                         reduction='batchmean',
                         log_target=True)
    return alpha * hard_loss + (1 - alpha) * soft_loss * (temp**2)


def train(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    epoches: int,
    dirname: str,
    max_to_keep=3,
    alpha=0.5,
    temperature=1.0,
):
    recoder = Recorder(dirname, max_to_keep=max_to_keep, moniter='test_acc')
    recoder.log('summary.txt', summary(student_model, INPUT_SIZE, verbose=0))

    optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           epoches,
                                                           eta_min=0.0)

    for epoch in range(epoches):
        print(f'Epoch {epoch + 1} / {epoches}')

        train_loss = MeanMetric().to(DEVICE)
        train_acc = Accuracy(task='multiclass',
                             num_classes=NUM_CLASSES).to(DEVICE)

        student_model.train()
        with tqdm(trainloader, desc='Train', dynamic_ncols=True) as pbar:
            for data, target in pbar:
                data, target = data.to(DEVICE), target.to(DEVICE)

                data = T.RandomResizedCrop(
                    INPUT_SIZE[-2:],
                    scale=(0.75, 1.0),
                )(data)

                target_prob = F.one_hot(target, NUM_CLASSES).float()

                data, target_prob = mixup(data, target_prob)

                with torch.no_grad():
                    teacher_output = teacher_model(data)

                output = student_model(data)
                loss = compute_loss(output, target_prob, teacher_output,
                                    temperature, alpha)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                train_loss.update(loss)
                train_acc.update(output, target)

                pbar.set_postfix_str(f'Loss: {train_loss.compute():.4f}, '
                                     f'Acc: {train_acc.compute():.4f}')

        scheduler.step()

        test_acc = Accuracy(task='multiclass',
                            num_classes=NUM_CLASSES).to(DEVICE)

        student_model.eval()
        with tqdm(testloader, desc='Test',
                  dynamic_ncols=True) as pbar, torch.no_grad():
            for data, target in pbar:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = student_model(data)

                test_acc.update(output, target)

                pbar.set_postfix_str(f'Acc: {test_acc.compute():.4f}')

        log_criterias = {
            'train_loss': train_loss.compute(),
            'train_acc': train_acc.compute(),
            'test_acc': test_acc.compute()
        }

        recoder.update(student_model.state_dict(), epoch + 1,
                       test_acc.compute(), log_criterias)

    student_model.load_state_dict(
        torch.load(recoder.best_filename, map_location=DEVICE))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # type: ignore

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_SIZE = (1, 3, 28, 28)

    NUM_CLASSES = 10
    LR = 0.001

    logdir = datetime.now().strftime('%Y%m%d-%H%M%S')

    trainloader, testloader = get_dataloaders()

    teacher_model = resnet50(num_classes=NUM_CLASSES)
    teacher_model.load_state_dict(torch.load('resnet50.pth'))
    teacher_model.to(DEVICE)
    teacher_model.eval()

    student_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    student_model.fc = torch.nn.Linear(512, NUM_CLASSES)
    student_model.to(DEVICE)

    n_params = summary(student_model, INPUT_SIZE, verbose=0).total_params

    train(teacher_model,
          student_model,
          trainloader,
          testloader,
          epoches=300,
          dirname=os.path.join(logdir, f'base'),
          alpha=0.1,
          temperature=10)

    remain_scheduler = torch.linspace(0.3, 0.5, 5).tolist()
    alpha_scheduler = torch.linspace(0.2, 0.8, 5).tolist()
    temp_scheduler = torch.linspace(8.0, 4.0, 5).tolist()

    prune_scheduler = zip(remain_scheduler, alpha_scheduler, temp_scheduler)

    prune_times = 0
    for remain_ratio, alpha, temp in prune_scheduler:
        prune_model(student_model, 1 - remain_ratio)
        prune_times += 1

        n_params = summary(student_model, INPUT_SIZE, verbose=0).total_params

        train(teacher_model,
              student_model,
              trainloader,
              testloader,
              epoches=300,
              dirname=os.path.join(logdir, f'prune_{prune_times}'),
              alpha=alpha,
              temperature=temp)

    if n_params > 100000:
        prune_model(student_model, n_params - 100000)
        prune_times += 1

        n_params = summary(student_model, INPUT_SIZE, verbose=0).total_params

        train(teacher_model,
              student_model,
              trainloader,
              testloader,
              epoches=300,
              dirname=os.path.join(logdir, f'prune_{prune_times}'),
              alpha=0.95,
              temperature=2)
