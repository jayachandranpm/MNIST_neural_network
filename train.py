import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN
from data import load_mnist_data_with_augmentation

def train_model(epochs=20, learning_rate=0.001):
    trainloader, testloader = load_mnist_data_with_augmentation()
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    return model

if __name__ == "__main__":
    train_model(epochs=20, learning_rate=0.001)
