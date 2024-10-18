
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

from dataset.animaldataset import AnimalDataset

# 전처리 
def load_transform(data_type='cifar'): 
    # 여기도 data_type에 맞춰 적용 
    if data_type == 'cifar': 
        # cifar10 이미지의 평균과 표준편차 
        CIFAR_MEAN = [0.491, 0.482, 0.447]
        CIFAR_STD = [0.247, 0.244, 0.262]

        transform = Compose([ToTensor(), 
                            Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)])
    elif data_type == 'animal': 
        # 평균과 표준편차 
        MEAN = [0.41429894, 0.40721159, 0.34098609]
        STD = [0.30540783, 0.29195736, 0.29756198]
        # 이미지 크기 변화도 넣어야 함 
        transform = Compose([Resize((32, 32)), 
                            ToTensor(), 
                            Normalize(mean=MEAN, std=STD)])
    return transform

# dataset 필요 
def load_datasets(transform, data_type='cifar'):
    # 역시 data_type에 맞춰 적용
    if data_type == 'cifar': 
        train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
    elif data_type == 'animal': 
        # 구현한 dataset class 불러오기 
        dataset = AnimalDataset(folder_path='../animal_data', transform=transform)
        # 이를 train, test로 나눠줘야 함 (random_split 함수 활용)
        from torch.utils.data import random_split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, val_size])
        
    return train_dataset, test_dataset

# dataloader 필요 
def load_dataloaders(batch_size, data_type): 
    # cifar면 기존 코드를 따라가고 
    if data_type == 'cifar': 
        transform = load_transform()
        train_dataset, test_dataset = load_datasets(transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # animal 이면 새롭게 만들어야 함 
    elif data_type == 'animal': 
        # 이미 구한 Mean, std 가지고 전처리 객체 가져와야 하고 
        # animal custom dataset을 가져와야 함
        # dataloader는 비슷하게 가져오면 됨 
        transform = load_transform(data_type)
        train_dataset, test_dataset = load_datasets(transform, data_type)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else : 
        raise ValueError('data_type은 cifar나 animal 둘 중 하나여야 함!')
    return train_loader, test_loader