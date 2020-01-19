import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import copy
from random import shuffle
from PIL import Image

"""
For now, using the 28x28 version requires having the unscaled USPS gallery saved in the same
directory as this script. You can find it here https://github.com/darshanbagul/USPS_Digit_Classification
"""

class USPSDataset28x28(data.Dataset):
    """Constructs the dataset of all available labelled USPS images"""
    def __init__(self, spec):
        self.root = spec['root']
        self.images = []
        self.labels = []
        self.transforms = spec['transforms']
        self.max_imgs_per_digit = spec['max_imgs_per_digit']

        for num in range(10):
            folder = os.path.join(self.root, 'Numerals', str(num))
            filenames = [os.path.join(folder, img_name) for img_name in os.listdir(folder)
                         if img_name.endswith('.png')]
            shuffle(filenames)

            imgs_processed = 0
            for img_file in filenames:
                # For some reason if not converted to B&W, there are grey patches in the tensor repr of image (Why?)
                img_tensor = self.transforms(Image.open(img_file).convert('1'))
                self.images.append(img_tensor)
                self.labels.append(num)
                imgs_processed += 1
                if self.max_imgs_per_digit and imgs_processed >= self.max_imgs_per_digit:
                    break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


std_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

# class ComposedTransforms:
#
#     usps_transf = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ])
#
#     mnist_transf = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ])


class FakeMNISTDataset(data.Dataset):
    def __init__(self, solver):
        self.G = copy.deepcopy(solver.G_UM)
        self.G = self.G.cpu()
        self.fake_mnist_imgs = []
        self.labels = []
        with torch.no_grad():
            for (usps_img, label) in solver.usps_test_loader.dataset:
                transformed_img = self.G(torch.unsqueeze(usps_img, dim=0))
                # output is a batch of size 1; remove batch dimension
                # i.e. shape [1, 1, 28, 28] --> [1, 28, 28]
                transformed_img = transformed_img.squeeze(dim=0)
                self.fake_mnist_imgs.append(transformed_img)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.fake_mnist_imgs[index], self.labels[index]


def get_loaders(config):
    # Get train loaders
    usps_spec = {'root': 'USPSdata',
                 'transforms': std_transform,
                 'max_imgs_per_digit': config.max_imgs_per_digit}
    usps_all_dataset = USPSDataset28x28(usps_spec)

    usps_train_size = int(config.train_prop * len(usps_all_dataset))
    usps_test_size = len(usps_all_dataset) - usps_train_size
    usps_train_dataset, usps_test_dataset = torch.utils.data.random_split(usps_all_dataset, [usps_train_size, usps_test_size])

    usps_train_loader = data.DataLoader(dataset=usps_train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers)
    usps_test_loader = data.DataLoader(dataset=usps_test_dataset,
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=config.num_workers)
    mnist_train_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                                     train=True,
                                                     download=True,
                                                     transform=std_transform)
    mnist_train_loader = data.DataLoader(dataset=mnist_train_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=config.num_workers)

    return usps_train_loader, mnist_train_loader, usps_test_loader


def get_fake_mnist_loader(solver):
    print("Initialising Fake MNIST DataLoader...", end=' ')
    fake_mnist_dataset = FakeMNISTDataset(solver)
    fake_mnist_loader = data.DataLoader(dataset=fake_mnist_dataset,
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=4)
    print("done")
    return fake_mnist_loader
