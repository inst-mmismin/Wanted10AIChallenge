import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = 0.001 
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 10. cpu, cuda 위치 바뀜 

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        super().__init__() # 1. 모델 초기화 과정에서 nn.Module 초기화 먼저 진행 
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.image_size * self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x) # 5. x의 shape 변경 
        return x

myMLP = MLP(image_size, hidden_size, num_classes).to(device) # 11. model을 gpu 위에 올려야 함 

train_mnist = MNIST(root='data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='data/mnist', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True) # 2. batch size 변수가 str 이었음 
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

optim = Adam(params=myMLP.parameters(), lr=lr) # 3. str -> float / 4. paramters 가 아니라 parameters() 이어야 함 

for epoch in range(total_epochs): 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device)

        output = myMLP(image)

        loss = loss_fn(output, label) # 6. Loss 객체 선언 해야함 / 7. output의 shape 안에 엄한값이 들어감 

        loss.backward()
        optim.step() # 8. step은 optimizer의 메소드 
        optim.zero_grad()

        if idx % 100 == 0 :  # 9. 나머지 값이 100이 아니라 0이어야 함 
            print(loss)
