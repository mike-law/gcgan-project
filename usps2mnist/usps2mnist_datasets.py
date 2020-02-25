import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import _pickle as cPickle
import gzip
import os


class USPSDataset(data.Dataset):
    # Num of Train = 7438, Num of Test 1860.
    def __init__(self, spec):
        self.filename = 'usps_28x28.pkl.gz'
        self.root = spec['root']
        # self.test_set_size = 0
        self.train_data, self.train_labels = self.load_samples()
        self.num_training_samples = len(self.train_labels)
        np.random.seed()
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
        images = data_set[0][0]
        labels = data_set[0][1]
        return images, labels


def get_loaders(config):
    std_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    usps_spec = {'root': './usps2mnist/data', 'transforms': std_transform, 'max_imgs': config.max_imgs}
    usps_dataset = USPSDataset(usps_spec)
    mnist_dataset = torchvision.datasets.MNIST(root='./usps2mnist/data/MNIST', train=True, download=True, transform=std_transform)
    usps_loader = data.DataLoader(dataset=usps_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    mnist_loader = data.DataLoader(dataset=mnist_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers)
    return usps_loader, mnist_loader