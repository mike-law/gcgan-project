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
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.model_locations = dict()
        self.model_locations['MNIST'] = config.pretrained_mnist_model
        self.accuracy = None
        self.losses_D = np.empty(0)
        self.losses_G = np.empty(0)
        # self.fake_mnist_buffer = None

        self.init_models()
        print("done")

    @abstractmethod
    def init_models(self):
        """Build generator(s) and discriminator(s)"""
        pass

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

    def get_test_visuals(self):
        with torch.no_grad():
            fig = plt.figure(figsize=(4, 3))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
            ax0 = plt.subplot(gs[0])
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
            ax1 = plt.subplot(gs[1])
            ax1.plot(self.losses_D, label="D loss")
            ax1.plot(self.losses_G, label="G loss")
            ax1.legend()
            fig.show()

    def test(self, fake_mnist_loader):
        # run a pretrained MNIST model over transformed images from self.usps_test_dataset
        if self.model_locations['MNIST'] is None:
            print("Pretrained MNIST model not given. Creating and training one now.")
            pretrained_mnist_model, self.model_locations['MNIST'] =\
                mnist_model.create_train_save(batch_size=32, num_epochs=4, save=True, timestamp=self.timestamp)
        else:
            pretrained_mnist_model = mnist_model.load_model(self.model_locations['MNIST'])
        print("Testing MNIST model against GAN...", end=' ')
        self.accuracy = mnist_model.get_test_accuracy(pretrained_mnist_model, fake_mnist_loader)
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