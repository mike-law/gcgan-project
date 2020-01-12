import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import usps_mnist_datasets_28x28 as um_datasets
import math
from datetime import datetime


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 24, kernel_size=5)
        self.fc1 = nn.Linear(384, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        y = F.relu(F.max_pool2d(self.conv2(y), 2))
        y = F.dropout(y, p=0.15)
        y = y.view(-1, 384)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=0.5)
        y = self.fc2(y)
        return y


def get_loaders(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                               train=True,
                                               download=True,
                                               transform=um_datasets.ComposedTransforms.mnist_transf)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)

    test_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                              train=False,
                                              download=True,
                                              transform=um_datasets.ComposedTransforms.mnist_transf)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)
    return train_loader, test_loader


def init_model(lr=0.004, momentum=0.9):
    model = Classifier()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)
    return model, optimizer, lr_scheduler


def train(model, optimizer, lr_scheduler, train_loader, num_epochs):
    model.train()
    n_batches = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
    for epoch in range(num_epochs):
        batch_count = 0
        for (img, labels) in train_loader:
            running_loss = 0.0
            if torch.cuda.is_available():
                img = img.cuda()
                labels = labels.cuda()
            pred = model(img)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_count + 1) % 200 == 0:
                print("Epoch {}, batch {:4d}/{}, avg loss per batch = {:.6f}".format(
                    epoch+1, batch_count+1, n_batches, running_loss / 200))
                lr_scheduler.step()
            batch_count += 1

    return model


def test(model, test_loader):
    model.eval()
    correct = 0
    total_loss = 0
    for (img, labels) in test_loader:
        img = img.cuda()
        labels = labels.cuda()
        output = model(img)
        total_loss += nn.CrossEntropyLoss()(output, labels)
        pred = torch.max(output.data, dim=1)[1]
        correct += torch.sum(torch.eq(pred, labels)).item()
    avg_loss = total_loss / len(test_loader.dataset)
    correct_prop = correct / len(test_loader.dataset) * 100
    print("{:.3f}% classified correctly, Average loss: {:.5f}".format(correct_prop, avg_loss))
    return correct_prop

def save_model(model):
    date_str = datetime.now().strftime("%y%m%d-%H%M%S")
    model_name = date_str + "-MNISTClassifier.pth"
    model_path = "./models/MNISTClassifier/" + model_name
    print("Saving model to {}".format(model_path))
    torch.save(model.state_dict(), model_path)