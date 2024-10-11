# 패키지 임포트 
import torch 
import torch.nn as nn 
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam

# Hparam 설정
batch_size = 100 
num_classes = 10
num_epochs = 10
# device 설정 
device='cuda' if torch.cuda.is_available() else 'cpu' 

## 데이터 처리하는 과정 (CIFAR10 활용)
# 전처리 
# cifar10 이미지의 평균과 표준편차 
CIFAR_MEAN = [0.491, 0.482, 0.447]
CIFAR_STD = [0.247, 0.244, 0.262]

transform = Compose([ToTensor(), 
                     Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)])

# dataset 필요 
train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)

# dataloader 필요 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

## 모델 만들기 ##
# 모델 설계도 만들기 
class myVGG(nn.Module): 
    def __init__(self, num_classes=10): 
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.out = nn.Sequential(
            nn.Linear(512, 100), 
            nn.ReLU(), 
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, num_classes)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
# 모델 생성에 사용이 될 Hparam 설정 
''' 상단에 전체 Hparam 적는 과정에 기입 '''
# 모델 객체를 생성 (설계도 + Hparam)
model = myVGG(num_classes=num_classes).to(device)

# Loss
loss_func = nn.CrossEntropyLoss()
# Optimizer 
optimizer = Adam(model.parameters(), lr=0.0001)

# 평가 함수 
def evaluate(model, test_loader): 
    model.eval()
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for images, labels in test_loader: 
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

best_acc = 0
# for loop를 돌면서 데이터를 불러오기 
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        images = images.to(device)
        labels = labels.to(device)

        # 불러온 데이터를 모델에 넣기 
        outputs = model(images)
        # 나온 출력물로 loss를 계산 
        loss = loss_func(outputs, labels)
        
        # Loss로 back prop 진행 
        loss.backward()
        # optimizer를 이용해 최적화를 진행 
        optimizer.step()
        optimizer.zero_grad()

        # 학습 중간에 
        if (i+1) % 100 == 0: 
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            # 평가를 진행해서 
            acc = evaluate(model, test_loader)
            # 성능이 좋으면 
            if acc > best_acc: 
                best_acc = acc
                # 저장을 진행 
                torch.save(model.state_dict(), f'./best_model.ckpt')
