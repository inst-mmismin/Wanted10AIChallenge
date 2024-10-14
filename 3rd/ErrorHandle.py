import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = '0.001'
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.image_size * self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

myMLP = MLP(image_size, hidden_size, num_classes)

train_mnist = MNIST(root='data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='data/mnist', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(dataset=train_mnist, batch_size='batch_size', shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size='batch_size', shuffle=True)

loss_fn = nn.CrossEntropyLoss

optim = Adam(params=myMLP.parameters, lr=lr)

for epoch in range(total_epochs): 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device)

        output = myMLP(image)

        loss = loss_fn(output, label)

        loss.backward()
        loss.step()
        optim.zero_grad()

        if idx % 100 == 100 : 
            print(loss)
