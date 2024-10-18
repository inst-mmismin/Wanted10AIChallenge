# 패키지를 임포트 
import torch 
import torch.nn as nn 
from PIL import Image
from torch.nn.functional import softmax

from utils.getModules import load_transform
from networks.vgg import myVGG

# device 설정 
device='cuda' if torch.cuda.is_available() else 'cpu' 

# 추론에 사용할 데이터를 준비 (전처리)
example_image_path = 'plane.png'
image = Image.open(example_image_path)
image = image.resize((32, 32))

# 학습에서 사용한 전처리 설정
transform = load_transform()

image = transform(image)
image = image.unsqueeze(0).to(device) 

## 학습이 완료된 최고의 모델을 준비하기 
num_classes = 10
model = myVGG(num_classes=num_classes)

# 속이 빈 모델에 학습된 모델의 wieght를 덮어 씌우기 
weight = torch.load('best_model.ckpt', map_location=device)
model.load_state_dict(weight)
model = model.to(device)

# 준비된 데이터를 모델에 넣기 
### 아래 코드를 작성해야 합니다. 
model.eval()
output = model(image)

## 결과를 분석 
# 결과를 사람이 이해할 수 있는 형태로 변환
probability = softmax(output, dim=1) # softmax : 확률의 형태로 변경 
values, indices = torch.max(probability, dim=1)
prob = values.item()*100
predict = indices.item()

print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')

# 모델의 추론 결과를 보고 객관적인 평가 내려보기 