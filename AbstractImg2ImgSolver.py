import GANLosses
import numpy as np
import torch
from abc import ABC, abstractmethod
import sys
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

class AbstractImg2ImgSolver(ABC):
    def __init__(self, config, A_loader, B_loader):
        self.path = "./testruns/" + datetime.now().strftime("%y%m%d-%H%M%S")
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
            iter_count + 1, n_iters, self.D_losses[-1], self.G_losses[-1], end=' '))
        print("avg D acc on real MNIST = {:.2f}%, avg D acc on fake MNIST = {:.2f}%".format(
            self.D_accuracies_real[-1], self.D_accuracies_fake[-1]))
        if self.D_losses[-1] < 0.005:
            print("Discriminator too strong! Try adjusting hyperparameters.")
            self.save_testrun()
            sys.exit(1)
        self.get_test_visuals(real_A_sample, fake_B_sample)

    def get_test_visuals(self, real_usps, fake_mnist, num_to_show=8):
        if self.latest_plot:
            plt.close(self.latest_plot)
        with torch.no_grad():
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])
            ax0 = fig.add_subplot(gs[:, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = ax1.twinx()
            ax3 = fig.add_subplot(gs[1, 1])

            # Image previews
            if self.config.input_nc == 3:
                inputs_joined = real_usps[:num_to_show].permute(0, 2, 3, 1).reshape(-1, self.config.image_size, 3)
                inputs_joined = (inputs_joined + 1) / 2
            elif self.config.input_nc == 1:
                inputs_joined = real_usps[:num_to_show].permute(0, 2, 3, 1).reshape(-1, self.config.image_size)
                inputs_joined = (inputs_joined + 1) / 2
            if self.config.output_nc == 3:
                outputs_joined = fake_mnist[:num_to_show].permute(0, 2, 3, 1).reshape(-1, self.config.image_size, 3)
                outputs_joined = (outputs_joined + 1) / 2
            elif self.config.output_nc == 1:
                outputs_joined = fake_mnist[:num_to_show].permute(0, 2, 3, 1).reshape(-1, self.config.image_size)
                outputs_joined = (outputs_joined + 1) / 2
            whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
            ax0.imshow(whole_grid)
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

    def save_testrun(self):
        os.mkdir(self.path)

        # save G_AB
        model_path = self.path + "/USPStoMNISTmodel.pth"
        torch.save(self.G_AB.state_dict(), model_path)
        print("Saved model as {}".format(model_path))

        # save config
        config_path = self.path + "/config.p"
        with open(config_path, "wb") as f:
            pickle.dump(self.config, file=f)

        # save text summary
        with open(self.path + "/summary.txt", "w+") as f:
            f.write(f"lambda_gan = {self.config.lambda_gan}\n")
            f.write(f"lambda_cycle = {self.config.lambda_cycle}\n")
            f.write(f"lambda_gc = {self.config.lambda_gc}\n")
            # f.write(f"lambda_reconst = {self.config.lambda_reconst}\n")
            # f.write(f"lambda_dist = {self.config.lambda_dist}\n")

            if self.config.lambda_gc > 0:
                f.write("GcGAN config:\n")
                f.write(f"geometry = {self.config.geometry}\n")
                if self.config.geometry in [4, 5, 6]:
                    f.write(f"noise_var = {self.config.noise_var}\n")
                f.write(f"separate_G = {self.config.separate_G}\n")
                f.write("\n")

            f.write("HYPERPARAMETERS\n")
            f.write(f"niter = {self.config.niter}\n")
            f.write(f"niter_decay = {self.config.niter_decay}\n")
            f.write(f"lr = {self.config.lr}\n")
            f.write(f"batch_size = {self.config.batch_size}\n\n")
            f.write("MODELS\n")
            f.write(f"g_conv_dim = {self.config.g_conv_dim}\n")
            f.write(f"d_conv_dim = {self.config.d_conv_dim}\n")

        # save diagnostic plots
        plot_path = self.path + "/plots.png"
        self.latest_plot.savefig(plot_path)

        # save sample image translations
        A_iter = iter(self.A_loader)
        with torch.no_grad():
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 3)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])

            for ax in [ax0, ax1, ax2]:
                real_A = next(A_iter).cuda()
                fake_B = self.G_AB(real_A)
                inputs_joined = real_A[:6].squeeze().view(-1, self.config.image_size)
                outputs_joined = fake_B[:6].squeeze().view(-1, self.config.image_size)
                whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
                ax.imshow(whole_grid, cmap='gray')
                ax.axis('off')

            sample_imgs_path = self.path + "/sample_translations.png"
            fig.savefig(sample_imgs_path)
