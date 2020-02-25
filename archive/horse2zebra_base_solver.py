import GANLosses
import numpy as np
import torch
from abc import ABC, abstractmethod
import sys
import matplotlib.pyplot as plt

class AbstractImg2ImgSolver(ABC):
    def __init__(self, config, A_loader, B_loader):
        self.A_loader = A_loader
        self.B_loader = B_loader
        self.config = config
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.target_real_label = 1.0
        self.target_fake_label = 0.0
        self.criterionGAN = GANLosses.GANLoss(config.use_lsgan, self.target_real_label, self.target_fake_label)

        # Log of results while training
        self.D_losses = np.empty(0)
        self.G_losses = np.empty(0)
        self.D_accuracies_fake = np.empty(0)
        self.D_accuracies_real = np.empty(0)
        self.latest_plot = None

        self.init_models()

    @abstractmethod
    def init_models(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_D_loss_and_bkwd(self, *args):
        pass

    @abstractmethod
    def get_G_loss_and_bkwd(self, *args):
        pass

    def report_results(self, iter_count, n_iters, loss_D_avg, loss_G_avg,
                       d_acc_real_mnist, d_acc_fake_mnist, real_A_sample, fake_B_sample):
        self.D_losses = np.append(self.D_losses, loss_D_avg)
        self.G_losses = np.append(self.G_losses, loss_G_avg)
        self.D_accuracies_fake = np.append(self.D_accuracies_fake, d_acc_fake_mnist)
        self.D_accuracies_real = np.append(self.D_accuracies_real, d_acc_real_mnist)
        print("{:04d} / {:04d} iters. avg loss_D = {:.5f}, avg loss_G = {:.5f},".format(
            iter_count + 1, n_iters, self.D_losses[-1], self.D_losses[-1], end=' '))
        print("avg D acc on real MNIST = {:.2f}%, avg D acc on fake MNIST = {:.2f}%".format(
            self.D_accuracies_real[-1], self.D_accuracies_fake[-1]))
        if self.D_losses[-1] < 0.005:
            print("Discriminator too strong! Try adjusting hyperparameters.")
            sys.exit(1)
        self.get_test_visuals(real_A_sample, fake_B_sample)

    def get_test_visuals(self, real_usps, fake_mnist, num_to_show=16):
        if self.latest_plot:
            plt.close(self.latest_plot)
        with torch.no_grad():
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 3])
            ax0 = fig.add_subplot(gs[:, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = ax1.twinx()
            ax3 = fig.add_subplot(gs[1, 1])

            # Image previews
            inputs_joined = real_usps[:num_to_show].squeeze().view(-1, self.config.image_size)
            outputs_joined = fake_mnist[:num_to_show].squeeze().view(-1, self.config.image_size)
            whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
            ax0.imshow(whole_grid, cmap='gray')
            ax0.axis('off')

            # Loss graphs
            xticks = np.arange(1, self.D_losses.size + 1) * 10
            ax1.plot(xticks, self.G_losses, 'b-', label="G loss")
            ax1.set_ylabel('G loss', color='b')
            ax2.plot(xticks, self.D_losses, 'r-', label="D loss")
            ax2.set_ylabel('D loss', color='r')

            # Discriminator accuracies on real and fake MNIST
            ax3.plot(xticks, self.D_accuracies_real, label="Real MNIST acc")
            ax3.plot(xticks, self.D_accuracies_fake, label="Fake MNIST acc")
            ax3.legend()
            ax3.grid(True)

            fig.show()
            self.latest_plot = fig

    ###
    # FILL IN REMAINING METHODS...
    ###
