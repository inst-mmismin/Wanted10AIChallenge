import torch 

# 평가 함수 
def evaluate(model, test_loader, device): 
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
    ### 아래 코드를 작성해야 합니다. 
    model.train()
    return correct / total
