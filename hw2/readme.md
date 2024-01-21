# Knowlege Distillation

## Environment

### Create environment

```bash
conda create --name hw2 python=3.8 -y
conda activate hw2

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install tqdm pandas torchmetrics torchinfo -c conda-forge -y
```

or

```bash
conda env create --file environment.yml
```

## Usage

### Train

```bash
python train.py
```

### Predict

```bash
python predict.py -w <weight_path>
```

or use pretraind weight

```bash
python predict.py -w pruned_resnet18.pth
```

## Method

### Data Augmentation

Apply two data augmentation methods sequentially:

1. RandomResizedCrop
2. mixup

## Training

Two stage training:

1. Knowlege Distillation
    - teacher model: ResNet50 with test accuracy 92.5%
    - student model: ResNet18

2. Repeatly prune weights then fine-tune.

## Prune Result

```yaml
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 10]                   --
├─Conv2d: 1-1                            [1, 64, 14, 14]           283
├─BatchNorm2d: 1-2                       [1, 64, 14, 14]           128
├─ReLU: 1-3                              [1, 64, 14, 14]           --
├─MaxPool2d: 1-4                         [1, 64, 7, 7]             --
├─Sequential: 1-5                        [1, 64, 7, 7]             --
│    └─BasicBlock: 2-1                   [1, 64, 7, 7]             --
│    │    └─Conv2d: 3-1                  [1, 64, 7, 7]             732
│    │    └─BatchNorm2d: 3-2             [1, 64, 7, 7]             128
│    │    └─ReLU: 3-3                    [1, 64, 7, 7]             --
│    │    └─Conv2d: 3-4                  [1, 64, 7, 7]             758
│    │    └─BatchNorm2d: 3-5             [1, 64, 7, 7]             128
│    │    └─ReLU: 3-6                    [1, 64, 7, 7]             --
│    └─BasicBlock: 2-2                   [1, 64, 7, 7]             --
│    │    └─Conv2d: 3-7                  [1, 64, 7, 7]             879
│    │    └─BatchNorm2d: 3-8             [1, 64, 7, 7]             128
│    │    └─ReLU: 3-9                    [1, 64, 7, 7]             --
│    │    └─Conv2d: 3-10                 [1, 64, 7, 7]             815
│    │    └─BatchNorm2d: 3-11            [1, 64, 7, 7]             128
│    │    └─ReLU: 3-12                   [1, 64, 7, 7]             --
├─Sequential: 1-6                        [1, 128, 4, 4]            --
│    └─BasicBlock: 2-3                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-13                 [1, 128, 4, 4]            1,681
│    │    └─BatchNorm2d: 3-14            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-15                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-16                 [1, 128, 4, 4]            3,216
│    │    └─BatchNorm2d: 3-17            [1, 128, 4, 4]            256
│    │    └─Sequential: 3-18             [1, 128, 4, 4]            505
│    │    └─ReLU: 3-19                   [1, 128, 4, 4]            --
│    └─BasicBlock: 2-4                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-20                 [1, 128, 4, 4]            3,654
│    │    └─BatchNorm2d: 3-21            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-22                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-23                 [1, 128, 4, 4]            3,775
│    │    └─BatchNorm2d: 3-24            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-25                   [1, 128, 4, 4]            --
├─Sequential: 1-7                        [1, 256, 2, 2]            --
│    └─BasicBlock: 2-5                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-26                 [1, 256, 2, 2]            7,345
│    │    └─BatchNorm2d: 3-27            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-28                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-29                 [1, 256, 2, 2]            18,286
│    │    └─BatchNorm2d: 3-30            [1, 256, 2, 2]            512
│    │    └─Sequential: 3-31             [1, 256, 2, 2]            1,324
│    │    └─ReLU: 3-32                   [1, 256, 2, 2]            --
│    └─BasicBlock: 2-6                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-33                 [1, 256, 2, 2]            3,471
│    │    └─BatchNorm2d: 3-34            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-35                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-36                 [1, 256, 2, 2]            3,529
│    │    └─BatchNorm2d: 3-37            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-38                   [1, 256, 2, 2]            --
├─Sequential: 1-8                        [1, 512, 1, 1]            --
│    └─BasicBlock: 2-7                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-39                 [1, 512, 1, 1]            15,261
│    │    └─BatchNorm2d: 3-40            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-42                 [1, 512, 1, 1]            7,509
│    │    └─BatchNorm2d: 3-43            [1, 512, 1, 1]            1,024
│    │    └─Sequential: 3-44             [1, 512, 1, 1]            4,255
│    │    └─ReLU: 3-45                   [1, 512, 1, 1]            --
│    └─BasicBlock: 2-8                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-46                 [1, 512, 1, 1]            8,240
│    │    └─BatchNorm2d: 3-47            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-49                 [1, 512, 1, 1]            6,583
│    │    └─BatchNorm2d: 3-50            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 1, 1]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 10]                   91
==========================================================================================
Total params: 100,000
Trainable params: 100,000
Non-trainable params: 0
Total mult-adds (M): 0.60
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.69
Params size (MB): 0.40
Estimated Total Size (MB): 1.10
==========================================================================================
```
