# 패키지 임포트 
import torch 
import torch.nn as nn 
from torch.optim import Adam

from utils.tools import evaluate
from utils.getModules import load_dataloaders
from networks.vgg import myVGG

# Hparam 설정
batch_size = 100 
num_classes = 10
num_epochs = 10
# device 설정 
device='cuda' if torch.cuda.is_available() else 'cpu' 

## 데이터 처리하는 과정 (CIFAR10 활용)
train_loader, test_loader = load_dataloaders(batch_size)

## 모델 만들기 ##
    
# 모델 생성에 사용이 될 Hparam 설정 
''' 상단에 전체 Hparam 적는 과정에 기입 '''
# 모델 객체를 생성 (설계도 + Hparam)
model = myVGG(num_classes=num_classes).to(device)

# Loss
loss_func = nn.CrossEntropyLoss()
# Optimizer 
optimizer = Adam(model.parameters(), lr=0.0001)


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
