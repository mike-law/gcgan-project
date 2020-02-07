import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import gzip
import _pickle as cPickle
import os
import copy
from random import shuffle
from PIL import Image

"""
The USPSDatasetA28x28 dataset creator requires having the unscaled USPS gallery saved in the same
directory as this script. You can find it here https://github.com/darshanbagul/USPS_Digit_Classification

The USPSDatasetB28x28 creator simply loads the images from a pkl.gz file
"""


class USPSDatasetA28x28(data.Dataset):
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


# pixel range: 0-1
class USPSDatasetB28x28(data.Dataset):
    # Num of Train = 7438, Num of Test 1860. Only using training set for this task.
    def __init__(self, spec):
        self.filename = 'usps_28x28.pkl.gz'
        # self.train = spec['train']
        self.root = spec['root']
        self.test_set_size = 0
        self.train_data, self.train_labels = self.load_samples()
        self.num_training_samples = len(self.train_labels)
        np.random.seed()
        # if self.train:
        total_num_samples = self.train_labels.shape[0]
        indices = np.arange(total_num_samples)
        np.random.shuffle(indices)
        self.train_data = self.train_data[indices[0:self.num_training_samples], ::]
        self.train_labels = self.train_labels[indices[0:self.num_training_samples]]

    def __getitem__(self, index):
        img, label = self.train_data[index, ::], self.train_labels[index]
        img = (img - 0.5) * 2  # scale to range -1:1
        return img, label

    def __len__(self):
        # if self.train:
        return self.num_training_samples
        # else:
        #     return self.test_set_size

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, 'rb')
        data_set = cPickle.load(f, encoding='latin1')
        f.close()
        # if self.train:
        images = data_set[0][0]
        labels = data_set[0][1]
        # else:
        #     images = data_set[1][0]
        #     labels = data_set[1][1]
        #     self.test_set_size = labels.shape[0]
        return images, labels


def get_loaders(config):
    # Get train loaders
    std_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    usps_spec = {'root': 'USPSdata',
                 'transforms': std_transform,
                 'max_imgs_per_digit': config.max_imgs_per_digit}
    usps_train_dataset = USPSDatasetB28x28(usps_spec)
    usps_train_loader = data.DataLoader(dataset=usps_train_dataset,
                                        batch_size=config.batch_size,
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
    return usps_train_loader, mnist_train_loader


# class FakeMNISTDataset(data.Dataset):
#     def __init__(self, solver):
#         self.G = copy.deepcopy(solver.G_UM)
#         self.G = self.G.cpu()
#         self.fake_mnist_imgs = []
#         self.labels = []
#         with torch.no_grad():
#             for (usps_img, label) in solver.usps_train_loader.dataset:
#                 transformed_img = self.G(torch.unsqueeze(usps_img, dim=0))
#                 # output is a batch of size 1; remove batch dimension
#                 # i.e. shape [1, 1, 28, 28] --> [1, 28, 28]
#                 transformed_img = transformed_img.squeeze(dim=0)
#                 self.fake_mnist_imgs.append(transformed_img)
#                 self.labels.append(label)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         return self.fake_mnist_imgs[index], self.labels[index]


# def get_fake_mnist_loader(solver):
#     print("Initialising Fake MNIST DataLoader...", end=' ')
#     fake_mnist_dataset = FakeMNISTDataset(solver)
#     fake_mnist_loader = data.DataLoader(dataset=fake_mnist_dataset,
#                                         batch_size=32,
#                                         shuffle=False,
#                                         num_workers=4)
#     print("done")
#     return fake_mnist_loader
