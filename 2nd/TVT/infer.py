# 패키지를 임포트 
import torch 
import torch.nn as nn 
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.nn.functional import softmax

# device 설정 
device='cuda' if torch.cuda.is_available() else 'cpu' 

# 추론에 사용할 데이터를 준비 (전처리)
example_image_path = 'test_image.jpeg'
image = Image.open(example_image_path)
image = image.resize((32, 32))

# 학습에서 사용한 전처리 설정
CIFAR_MEAN = [0.491, 0.482, 0.447]
CIFAR_STD = [0.247, 0.244, 0.262]

transform = Compose([
    ToTensor(),
    Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
])

image = transform(image)
image = image.unsqueeze(0).to(device) 

## 학습이 완료된 최고의 모델을 준비하기 
# 설계도 + Hparam 모델 껍대기 만들기 
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
    
num_classes = 10
model = myVGG(num_classes=num_classes)

# 속이 빈 모델에 학습된 모델의 wieght를 덮어 씌우기 
weight = torch.load('best_model.ckpt', map_location=device)
model.load_state_dict(weight)
model = model.to(device)

# 준비된 데이터를 모델에 넣기 
output = model(image)

## 결과를 분석 
# 결과를 사람이 이해할 수 있는 형태로 변환
probability = softmax(output, dim=1) # softmax : 확률의 형태로 변경 
values, indices = torch.max(probability, dim=1)
prob = values.item()*100
predict = indices.item()

print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

# 모델의 추론 결과를 보고 객관적인 평가 내려보기 