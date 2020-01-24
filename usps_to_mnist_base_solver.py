import torch
import mnist_model
import GANLosses
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABC, abstractmethod
import pickle
import numpy as np


class AbstractSolver(ABC):

    def __init__(self, config, usps_train_loader, mnist_train_loader):
        # May need to refactor some of these so that they make sense in the class hierarchy.
        print("Initialising solver...", end=' ')
        self.timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        self.usps_train_loader = usps_train_loader
        self.mnist_train_loader = mnist_train_loader
        self.config = config
        self.target_real_label = 1.0
        self.target_fake_label = 0.0
        self.criterionGAN = GANLosses.GANLoss(config.use_lsgan, self.target_real_label, self.target_fake_label)
        self.criterionReconst = GANLosses.ReconstLoss() if self.config.lambda_reconst > 0 else None
        # self.criterionDist = GANLosses.DistanceLoss() if self.config.lambda_dist > 0 else None
        # if self.criterionDist:
        #     self.mean_dist_usps, self.stdev_dist_usps = self.get_mean_and_stdev_dist(usps_train_loader)
        #     self.mean_dist_mnist, self.stdev_dist_mnist = self.get_mean_and_stdev_dist(mnist_train_loader)
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.accuracy = None
        self.avg_losses_D = np.empty(0)
        self.avg_losses_G = np.empty(0)
        self.classifier_accuracies = np.empty(0)
        self.D_accuracies_fake = np.empty(0)
        self.D_accuracies_real = np.empty(0)
        self.latest_plot = None
        # Set up MNIST classifier model
        if self.config.pretrained_mnist_model:
            self.pretrained_mnist_model = mnist_model.load_model(self.config.pretrained_mnist_model)
        else:
            print("Pretrained MNIST model not given. Creating and training one now.")
            self.pretrained_mnist_model = mnist_model.create_and_train(batch_size=32, num_epochs=4)  #, save=True, timestamp=self.timestamp)
        self.init_models()
        print("done")

    @abstractmethod
    def init_models(self):
        """Build generator(s) and discriminator(s)"""
        pass

    # def get_mean_and_stdev_dist(self, dataloader):
    #     """ For distanceGAN. Returns the mean and stdev of pairwise distances in given dataloader """
    #     dataiter = iter(dataloader)
    #     # dist_fn = nn.L1Loss(reduction='sum')
    #     all_imgs = None # massive batch containing all images
    #     for img_batch, _ in dataiter:
    #         all_imgs = img_batch if all_imgs is None else torch.cat((all_imgs, img_batch))
    #     distance_sum = 0.0
    #     distance_squared_sum = 0.0
    #     for i in range(len(all_imgs) - 1):
    #         for j in range(i + 1, len(all_imgs)):
    #             distance = torch.sum(torch.abs(all_imgs[i] - all_imgs[j]))
    #             distance_sum += distance
    #             distance_squared_sum += distance ** 2
    #             print(i, j, distance)
    #     mean_dist = distance_sum / (len(all_imgs) ** 2)
    #     var_dist = distance_squared_sum / (len(all_imgs) ** 2) - mean_dist ** 2
    #     stdev_dist = np.sqrt(var_dist)
    #     return mean_dist, stdev_dist

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def backward_D(self, *args):
        """ Calculates the disciminator loss and backpropagates to the parameters. """
        pass

    @abstractmethod
    def backward_G(self, *args):
        """ Calculates the generator loss and backpropagates to the parameters. """
        pass

    def get_train_accuracy(self, fake_mnist, labels):
        # Returns the number of correctly labelled fake_mnist images (NOT the proportion/percentage)
        self.pretrained_mnist_model.eval()
        labels = labels.cuda()
        output = self.pretrained_mnist_model(fake_mnist)
        pred = torch.max(output.data, dim=1)[1]
        correct = torch.sum(torch.eq(pred, labels)).item()
        return correct

    def get_test_visuals(self, real_usps, fake_mnist, num_to_show=16):
        with torch.no_grad():
            fig = plt.figure(figsize=(4, 6))
            gs = fig.add_gridspec(3, 2, width_ratios=[1, 4], height_ratios=[2, 2, 1])
            ax0 = fig.add_subplot(gs[:, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 1])

            # Image previews
            inputs_joined = real_usps[:num_to_show].squeeze().view(-1, self.config.image_size)
            outputs_joined = fake_mnist[:num_to_show].squeeze().view(-1, self.config.image_size)
            whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
            ax0.imshow(whole_grid, cmap='gray')
            ax0.axis('off')

            # Loss graphs
            xticks = np.arange(1, self.avg_losses_D.size + 1) * 10
            ax1.plot(xticks, self.avg_losses_D, label="D loss")
            ax1.plot(xticks, self.avg_losses_G, label="G loss")
            ax1.legend()
            ax1.grid(True)

            # Discriminator accuracies on real and fake MNIST
            ax2.plot(xticks, self.D_accuracies_real, label="Real MNIST acc")
            ax2.plot(xticks, self.D_accuracies_fake, label="Fake MNIST acc")
            ax2.legend()
            ax2.grid(True)

            # Training accuracy
            ax3.plot(xticks, self.classifier_accuracies, label="% classification acc")
            ax3.legend()
            ax3.grid(True)

            fig.show()
            self.latest_plot = fig

    def test(self, fake_mnist_loader):
        # run pretrained MNIST model over transformed images from self.usps_test_dataset
        print("Testing MNIST model against GAN...", end=' ')
        self.pretrained_mnist_model.eval()
        correct = 0
        for (img, labels) in fake_mnist_loader:
            img = img.cuda()
            labels = labels.cuda()
            output = self.pretrained_mnist_model(img)
            pred = torch.max(output.data, dim=1)[1]
            correct += torch.sum(torch.eq(pred, labels)).item()
        self.accuracy = correct / len(fake_mnist_loader.dataset) * 100
        print("done")
        print("{:.3f}% of generated digits classified correctly".format(self.accuracy))

    def save_testrun(self, path):
        # save G_UM
        model_path = path + "/USPStoMNISTmodel.pth"
        torch.save(self.G_UM.state_dict(), model_path)
        print("Saved model as {}".format(model_path))

        # save config
        config_path = path + "/config.p"
        with open(config_path, "wb") as f:
            pickle.dump(self.config, file=f)

        # save text summary
        with open(path + "/summary.txt", "w+") as f:
            f.write(f"lambda_gan = {self.config.lambda_gan}\n")
            f.write(f"lambda_cycle = {self.config.lambda_cycle}\n")
            f.write(f"lambda_gc = {self.config.lambda_gc}\n")
            f.write(f"lambda_reconst = {self.config.lambda_reconst}\n")
            f.write(f"lambda_dist = {self.config.lambda_dist}\n")
            f.write(f"classification accuracy = {self.classifier_accuracies[-1]}%\n\n")
            f.write("HYPERPARAMETERS\n")
            f.write(f"niter = {self.config.niter}\n")
            f.write(f"niter_decay = {self.config.niter_decay}\n")
            f.write(f"lr = {self.config.lr}\n")
            f.write(f"batch_size = {self.config.batch_size}\n\n")
            f.write("MODELS\n")
            f.write(f"g_conv_dim = {self.config.g_conv_dim}\n")
            f.write(f"d_conv_dim = {self.config.d_conv_dim}\n")

        # save diagnostic plots
        plot_path = path + "/plots.png"
        self.latest_plot.savefig(plot_path)

        # save sample image translations
        usps_iter = iter(self.usps_train_loader)
        with torch.no_grad():
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(1, 3)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])

            for ax in [ax0, ax1, ax2]:
                real_usps, _ = next(usps_iter)
                real_usps = real_usps.cuda()
                fake_mnist = self.G_UM(real_usps)
                inputs_joined = real_usps[:6].squeeze().view(-1, self.config.image_size)
                outputs_joined = fake_mnist[:6].squeeze().view(-1, self.config.image_size)
                whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
                ax.imshow(whole_grid, cmap='gray')
                ax.axis('off')

            sample_imgs_path = path + "/sample_translations.png"
            fig.savefig(sample_imgs_path)
