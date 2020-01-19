import torch
import torch.nn as nn
import mnist_model
import networks
import GANLosses
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from abc import ABC, abstractmethod
import pickle
import numpy as np


class AbstractSolver(ABC):

    def __init__(self, config, usps_train_loader, mnist_train_loader, usps_test_loader):
        # May need to refactor some of these so that they make sense in the class hierarchy.
        print("Initialising solver...", end=' ')
        self.timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        self.usps_train_loader = usps_train_loader
        self.mnist_train_loader = mnist_train_loader
        self.usps_test_loader = usps_test_loader
        self.usps_test_iter = iter(self.usps_test_loader)
        self.config = config
        self.criterionGAN = GANLosses.GANLoss(config.use_lsgan)
        self.criterionReconst = GANLosses.ReconstLoss() if self.config.lambda_reconst > 0 else None
        # self.criterionDist = GANLosses.DistanceLoss() if self.config.lambda_dist > 0 else None
        # if self.criterionDist:
        #     self.mean_dist_usps, self.stdev_dist_usps = self.get_mean_and_stdev_dist(usps_train_loader)
        #     self.mean_dist_mnist, self.stdev_dist_mnist = self.get_mean_and_stdev_dist(mnist_train_loader)
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.accuracy = None
        self.avg_losses_D = np.empty(0)
        self.avg_losses_G = np.empty(0)
        self.train_accuracy_record = np.empty(0)
        self.model_locations = dict()
        self.model_locations['MNIST'] = config.pretrained_mnist_model
        # self.fake_mnist_buffer = None
        # Set up MNIST classifier model
        if self.model_locations['MNIST'] is None:
            print("Pretrained MNIST model not given. Creating and training one now.")
            self.pretrained_mnist_model, self.model_locations['MNIST'] =\
                mnist_model.create_train_save(batch_size=32, num_epochs=4, save=True, timestamp=self.timestamp)
        else:
            self.pretrained_mnist_model = mnist_model.load_model(self.model_locations['MNIST'])

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

    def get_test_visuals(self):
        with torch.no_grad():
            fig = plt.figure(figsize=(4, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[3, 2])
            ax0 = fig.add_subplot(gs[:, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])

            # Image previews
            try:
                usps_inputs = next(self.usps_test_iter)[0].cuda()
            except StopIteration:
                # Reached end of batch, restart iterator
                self.usps_test_iter = iter(self.usps_test_loader)
                usps_inputs = next(self.usps_test_iter)[0].cuda()
            mnist_outputs = self.G_UM(usps_inputs)
            # top row: original images (usps inputs)
            inputs_joined = usps_inputs.squeeze().view(-1, self.config.image_size)
            # bottom row: transformed images (mnist-ish outputs)
            outputs_joined = mnist_outputs.squeeze().view(-1, self.config.image_size)
            whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
            ax0.imshow(whole_grid, cmap='gray')
            ax0.axis('off')

            # Loss graphs
            xticks = np.arange(1, self.avg_losses_D.size + 1) * 10
            ax1.plot(xticks, self.avg_losses_D, label="D loss")
            ax1.plot(xticks, self.avg_losses_G, label="G loss")
            ax1.legend()
            ax1.grid(True)

            # Training accuracy
            ax2.plot(xticks, self.train_accuracy_record, label="training acc.")
            ax2.legend()
            ax2.grid(True)
            fig.show()

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

    def save_models(self):
        # later extend to saving all of G_UM, G_gc_UM, etc.
        model_path = "./models/USPStoMNIST/" + self.timestamp + "-USPStoMNISTmodel.pth"
        torch.save(self.G_UM.state_dict(), model_path)
        print("Saved model as {}".format(model_path))
        self.model_locations['G_UM'] = model_path

    def save_testrun(self):
        # save summary of hyperparameters, model created, and its accuracy on the given MNIST model
        summary = {'config': self.config,
                   'G_path': self.model_locations['G_UM'],
                   'MNIST_model': self.model_locations['MNIST'],
                   'accuracy': self.accuracy}
        testrun_file = "./testruns/" + self.timestamp + ".p"
        with open(testrun_file, "wb") as f:
            pickle.dump(summary, file=f)
        print("Saved test run summary as {}".format(testrun_file))