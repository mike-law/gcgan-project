import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image, ImageOps

"""
For now, using the 28x28 version requires having the unscaled USPS gallery saved in the same
directory as this script. You can find it here https://github.com/darshanbagul/USPS_Digit_Classification
"""

class get_usps_dataset_28x28(data.Dataset):
    def __init__(self, spec):
        self.mode = spec['mode'] # either 'train' or 'test'
        self.root = spec['root']
        self.images = None
        self.labels = None
        self.transforms = spec['transforms']
        self.max_imgs_per_num = spec['max_imgs_per_num']

        if self.mode == 'train':
            for num in range(10):
                folder = os.path.join(self.root, 'Numerals', str(num))
                filenames = [os.path.join(folder, img_name) for img_name in os.listdir(folder)
                             if img_name.endswith('a.png')] # only get the 'a' images else too many

                imgs_processed = 0
                for img_file in filenames:
                    img_tensor = self.transforms(Image.open(img_file))
                    self.images = img_tensor if self.images is None else torch.cat((self.images, img_tensor))
                    self.labels = torch.tensor([num]) if self.labels is None else \
                        torch.cat((self.labels, torch.tensor([num])))
                    imgs_processed += 1
                    if self.max_imgs_per_num and imgs_processed >= self.max_imgs_per_num:
                        break

        elif self.mode == 'test':
            # Give test datasets
            folder = os.path.join(self.root, 'Test')
            filenames = [os.path.join(folder, img_name) for img_name in os.listdir(folder)
                         if img_name.endswith('.png')]
            for img_file in filenames:
                img_tensor = self.transforms(Image.open(img_file))
                self.images = img_tensor if self.images is None else torch.cat((self.images, img_tensor))
            # Dummy -1 labels
            self.labels = torch.full(size=(len(filenames),), fill_value=-1)

        else:
            print("Invalid mode, must be 'train' or 'test'")

    def __len__(self):
        return self.images.size()[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class UnsqueezeSingleChannel(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.unsqueeze(img, dim=0)


class ComposedTransforms:

    usps_transf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        UnsqueezeSingleChannel()  # Add extra dimension for the single channel
    ])

    mnist_transf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_loaders(config):
    usps_spec = {'mode': 'train',
                 'root': 'USPSdata',
                 'transforms': ComposedTransforms.usps_transf,
                 'max_imgs_per_num': 220}

    usps_dataset = get_usps_dataset_28x28(usps_spec)
    usps_loader = data.DataLoader(dataset=usps_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)

    mnist_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                               train=True,
                                               download=True,
                                               transform=ComposedTransforms.mnist_transf)

    mnist_loader = data.DataLoader(dataset=mnist_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers)
    return usps_loader, mnist_loader


def get_usps_test_loader(batch_size):
    usps_spec = {'mode': 'test',
                 'root': 'USPSdata',
                 'transforms': ComposedTransforms.usps_transf,
                 'max_imgs_per_num': None}
    usps_test_dataset = get_usps_dataset_28x28(usps_spec)
    usps_test_loader = data.DataLoader(dataset=usps_test_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    return usps_test_loader
