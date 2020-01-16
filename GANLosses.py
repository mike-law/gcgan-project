import torch
import torch.nn as nn
from torch.autograd import Variable

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.base_loss(input, target_tensor)


class GCLoss(nn.Module):
    def __init__(self):
        super(GCLoss, self).__init__()
        self.base_loss = nn.L1Loss()

    def __call__(self, target, f_target, f, f_inv):
        return self.base_loss(f_inv(f_target), target) + self.base_loss(f(target), f_target)

class ReconstLoss(nn.Module):
    def __init__(self):
        super(ReconstLoss, self).__init__()
        self.base_loss = nn.L1Loss()

    def __call__(self, g_target, target):
        return self.base_loss(g_target, target)


class CycleLoss(nn.Module):
    def __init__(self):
        super(CycleLoss, self).__init__()
        self.base_loss = nn.L1Loss()

    def __call__(self, real_A, real_B, fake_A, fake_B, G_AB, G_BA):
        return self.base_loss(G_BA(fake_B), real_A) + self.base_loss(G_AB(fake_A), real_B)