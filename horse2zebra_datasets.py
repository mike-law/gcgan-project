import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from random import shuffle


class Horse2ZebraDataset(data.Dataset):
    def __init__(self, spec):
        # Spec: dict with keys 'root', 'transforms', 'max_imgs'
        self.root = spec['root']
        self.transforms = spec['transforms']
        self.max_imgs = spec['max_imgs']
        self.filenames = [os.path.join(spec['root'], img_name) for img_name in os.listdir(self.root)]
        if self.max_imgs:
            shuffle(self.filenames)
            self.filenames = self.filenames[:self.max_imgs]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img


def get_loaders(config):
    std_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    horse_spec = {'root': './horse2zebra/trainA', 'transforms': std_transform, 'max_imgs': config.max_imgs}
    zebra_spec = {'root': './horse2zebra/trainB', 'transforms': std_transform, 'max_imgs': config.max_imgs}
    horse_dataset = Horse2ZebraDataset(horse_spec)
    zebra_dataset = Horse2ZebraDataset(zebra_spec)
    horse_loader = data.DataLoader(dataset=horse_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers)
    zebra_loader = data.DataLoader(dataset=zebra_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers)
    return horse_loader, zebra_loader