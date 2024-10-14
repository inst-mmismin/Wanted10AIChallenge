
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

# 전처리 
def load_transform(): 
    # cifar10 이미지의 평균과 표준편차 
    CIFAR_MEAN = [0.491, 0.482, 0.447]
    CIFAR_STD = [0.247, 0.244, 0.262]

    transform = Compose([ToTensor(), 
                        Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)])
    return transform

# dataset 필요 
def load_datasets(transform):
    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# dataloader 필요 
def load_dataloaders(batch_size): 
    transform = load_transform()
    train_dataset, test_dataset = load_datasets(transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader