import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import usps_mnist_datasets_28x28 as um_datasets
import math


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
                                               transform=um_datasets.std_transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)

    test_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                              train=False,
                                              download=True,
                                              transform=um_datasets.std_transform)
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


def create_and_train(batch_size=32, num_epochs=4):  #, save=True, timestamp="undefined_time"):
    print("MNIST model: initialising DataLoaders...", end=' ')
    train_loader, test_loader = get_loaders(batch_size)
    print("done")
    print("MNIST model: initialising model...", end=' ')
    pretrained_model, optimizer, lr_scheduler = init_model()
    print("done")
    print("---------- MNIST model: Training model ----------")
    pretrained_model = train(pretrained_model, optimizer, lr_scheduler, train_loader, num_epochs)
    print("---------- MNIST model: Finished training ----------")
    # model_path = ""
    # if save:
    #     model_path += "./models/MNISTClassifier/" + timestamp + "-MNISTClassifier.pth"
    #     torch.save(pretrained_model.state_dict(), model_path)
    #     print("Saved model as {}".format(model_path))
    return pretrained_model  #, model_path


def load_model(model_file):
    model = Classifier()
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    return model