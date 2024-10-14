import torch 
from PIL import Image
from torch.nn.functional import softmax

from utils.getModules import load_transform
from networks.vgg import myVGG

device='cuda' if torch.cuda.is_available() else 'cpu' 

example_image_path = 'plane.png'
# 이미지 읽고 
# 크기 변환 

transform = load_transform()

# 전처리 진행 

num_classes = 10
model = myVGG(num_classes=num_classes)

weight = torch.load('best_model.ckpt', map_location=device)
model.load_state_dict(weight)
model = model.to(device)

model.eval()
output = model(image)

# 추론 결과 만들어내기 

print(f'해당 이미지를 보고 딥러닝 모델은 {prob:.2f}% 의 확률로 {predict}이라고 대답했다.')
