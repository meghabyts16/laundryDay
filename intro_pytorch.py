import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    set = ""
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if (training == True):
        set=datasets.FashionMNIST('./data',train=training, download=True,transform=custom_transform)
    else:
        set=datasets.FashionMNIST('./data', train=training, transform=custom_transform)
    loader = torch.utils.data.DataLoader(set, batch_size=64, shuffle=True)
    return loader

def build_model():
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T=5):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        running_loss = 0.0
        num_correct = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            num_correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * num_correct / len(train_loader.dataset)
        print(f"Train Epoch: {epoch} Accuracy: {num_correct}/{len(train_loader.dataset)} ({epoch_acc:.2f}%) Loss: {epoch_loss:.3f}")

def evaluate_model(model, test_loader, criterion, show_loss = True):
    total = 0
    correct = 0
    testing_loss = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            testing_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += inputs.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    testing_loss /= len(test_loader.dataset)

    if show_loss == True:
        print(f"Average loss: {testing_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    model.eval()
    
    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))
        prob = F.softmax(logits, dim=1)
        top_probs, top_labels = prob.topk(k=3, dim=1)
        
        print(f"{class_names[top_labels[0][0]]}: {top_probs[0][0]*100:.2f}%")
        print(f"{class_names[top_labels[0][1]]}: {top_probs[0][1]*100:.2f}%")
        print(f"{class_names[top_labels[0][2]]}: {top_probs[0][2]*100:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    N=10
    test_images = torch.randn(N, 1, 28, 28)
    predict_label(model, test_images, 1)

