import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image

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
                             if img_name.endswith('.png')]

                imgs_processed = 0
                for img_file in filenames:
                    # For some reason if not converted to B&W, there are grey patches in the tensor repr of image (Why?)
                    img_tensor = self.transforms(Image.open(img_file).convert('1'))
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
                # For some reason if not converted to B&W, there are grey patches in the tensor repr of image (Why?)
                img_tensor = self.transforms(Image.open(img_file).convert('1'))
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


def get_loaders(config):
    # Get train loaders
    usps_train_spec = {'mode': 'train',
                       'root': 'USPSdata',
                       'transforms': ComposedTransforms.usps_transf,
                       'max_imgs_per_num': 600}

    usps_train_dataset = get_usps_dataset_28x28(usps_train_spec)
    usps_train_loader = data.DataLoader(dataset=usps_train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers)

    mnist_train_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                                     train=True,
                                                     download=True,
                                                     transform=ComposedTransforms.mnist_transf)

    mnist_train_loader = data.DataLoader(dataset=mnist_train_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=config.num_workers)

    # Get USPS train loader
    usps_test_spec = {'mode': 'test',
                      'root': 'USPSdata',
                      'transforms': ComposedTransforms.usps_transf,
                      'max_imgs_per_num': None}
    usps_test_dataset = get_usps_dataset_28x28(usps_test_spec)
    usps_test_loader = data.DataLoader(dataset=usps_test_dataset,
                                       batch_size=4,
                                       shuffle=True)

    return usps_train_loader, mnist_train_loader, usps_test_loader
