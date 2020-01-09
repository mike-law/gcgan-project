import torch
import torch.optim as optim
import torch.nn as nn
import itertools
import networks
import matplotlib.pyplot as plt

"""
USPS -> MNIST solver
"""


class Solver(object):
    def __init__(self, config, usps_train_loader, mnist_train_loader, usps_test_loader):
        self.usps_train_loader = usps_train_loader
        self.mnist_train_loader = mnist_train_loader
        self.usps_test_loader = usps_test_loader
        self.usps_test_iter = iter(self.usps_test_loader)
        self.image_size = config.image_size
        self.isTrain = config.train
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.criterionGAN = networks.GANLoss(use_lsgan=True)
        self.criterionGc = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.config = config
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator"""

        # Nets
        # let's use a single network for both generators
        self.G_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG='resnet_9blocks', norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        # self.G_gc_UM = networks.define_G(input_nc=1, output_nc=1, ngf=32, which_model_netG='resnet_6blocks',
        #                                  norm='batch', init_type='normal', gpu_ids=self.gpu_ids)

        if self.isTrain:
            # two separate discriminators
            self.D_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim, which_model_netD='basic',
                                         n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal',
                                         gpu_ids=self.gpu_ids)
            # self.D_gc_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim, which_model_netD='basic',
            #                                 n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal',
            #                                 gpu_ids=self.gpu_ids)

        # Optimisers
        self.G_optim = optim.Adam(self.G_UM.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        # self.G_optim = optim.Adam(itertools.chain(self.G_UM.parameters(), self.G_gc_UM.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizers = [self.G_optim]

        if self.isTrain:
            # a single optimiser for both discriminators (hence need a combined loss and .backward() computes
            # gradients for both networks simultaneously
            self.D_optim = optim.Adam(self.D_M.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            # self.D_optim = optim.Adam(itertools.chain(self.D_M.parameters(), self.D_gc_M.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizers.append(self.D_optim)

        # Schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, self.config))

    def get_gc_loss(self, AB, AB_gc): # AB = G(X), AB_gc = G~(f(X)).
        """Calculates the Gc loss which is a sum of two L1 losses"""
        loss = 0.0
        AB_gt = self.rot90(tensor=AB, direction=0)
        # Ideally AB_gt and AB_gc are similar
        loss += self.criterionGc(AB_gc, AB_gt)

        AB_gc_gt = self.rot90(tensor=AB_gc, direction=1)
        # Ideally AB_gc_gt and AB_gc are similar
        loss += self.criterionGc(AB_gc, AB_gc_gt)

        return loss

    def rot90(self, tensor, direction): # 0 = clockwise, 1 = counterclockwise
        t = torch.transpose(tensor, 2, 3).cuda()
        inv_idx = torch.arange(self.image_size-1, -1, -1).long().cuda()

        if direction == 0:
            t = torch.index_select(t, 3, inv_idx)
        elif direction == 1:
            t = torch.index_select(t, 2, inv_idx)

        return t

    def train(self):
        print('----------- Training has started. -----------')
        n_iters = self.config.niter + self.config.niter_decay
        iter_count = 0
        while True:
            usps_train_iter = iter(self.usps_train_loader)
            mnist_train_iter = iter(self.mnist_train_loader)
            for usps_batch, mnist_batch in zip(usps_train_iter, mnist_train_iter):
                usps, u_labels = usps_batch
                mnist, m_labels = mnist_batch
                usps = usps.cuda()
                mnist = mnist.cuda()

                # Generate
                mnist_rot = None # self.rot90(mnist, 0)
                fake_mnist = self.G_UM.forward(usps)
                fake_mnist_rot = None # self.G_UM.forward(self.rot90(usps, 0))

                # what do D and D_gc think?
                pred_d_fake = self.D_M(fake_mnist)
                pred_d_gc_fake = None # self.D_gc_M(fake_mnist_rot)
                pred_d_real = self.D_M(mnist)
                pred_d_gc_real = None # self.D_gc_M(mnist_rot)

                # if (iter_count + 1) % 2 == 0:
                # backward D and D_gc. Use a single loss function since D_optim has params of both
                self.backward_D(pred_d_fake, pred_d_gc_fake, pred_d_real, pred_d_gc_real)

                if (iter_count + 1) % 2 == 0:
                # backward G (and hence G_gc at the same time) (use the same batch as above)
                    self.backward_G(fake_mnist, fake_mnist_rot, mnist, mnist_rot, pred_d_fake, pred_d_gc_fake)

                # update learning rates
                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 10 == 0:
                    print("{:04d} of {:04d} iterations. loss_D = {:.5f}, loss_G = {:.5f}".format(
                        iter_count + 1, n_iters, self.loss_D, self.loss_G))
                    self.get_test_visuals()

                iter_count += 1
                # if all iterations done, break out of both loops
                if iter_count >= n_iters:
                    break
            if iter_count >= n_iters:
                break

        # DONE
        print('----------- Finished training -----------')

    def backward_D(self, pred_d_fake, pred_d_gc_fake, pred_d_real, pred_d_gc_real):
        self.D_optim.zero_grad()

        # D trying to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss_d = (self.criterionGAN(pred_d_fake.cpu(), False) + self.criterionGAN(pred_d_real.cpu(), True))  # * 0.5
        # loss_d_gc = 0.5 * (self.criterionGAN(pred_d_gc_fake.cpu(), False) + self.criterionGAN(pred_d_gc_real.cpu(), True))

        loss = loss_d # + loss_d_gc
        loss = loss.cuda()
        loss.backward(retain_graph=True)
        self.D_optim.step()

        self.loss_D = loss.cpu()

    def backward_G(self, fake_mnist, fake_mnist_rot, real_mnist, real_mnist_rot, pred_d_fake, pred_d_gc_fake):
        #, pred_d_real, pred_d_gc_real):
        self.G_optim.zero_grad()

        # GAN loss
        loss_g_gan = self.criterionGAN(pred_d_fake.cpu(), True)  # * 0.5
        # loss_g_gan += self.criterionGAN(pred_d_gc_fake.cpu(), True) * 0.5
        loss = loss_g_gan.cuda()

        # if self.config.lambda_gc > 0:
        #     # GC loss
        #     loss_g_gc = self.criterionGc(self.rot90(fake_mnist_rot, 1), fake_mnist)
        #     loss_g_gc += self.criterionGc(self.rot90(fake_mnist, 0), fake_mnist_rot)
        #     loss_g_gc *= self.config.lambda_gc
        #     loss += loss_g_gc

        # if self.config.use_reconst_loss:
        #     # Id loss (only penalises G)
        #     # Based on the idea that G(real_mnist) should be mnist, and G(real_mnist_rot) should be mnist_rot.
        #     loss_g_idt = 0.5 * self.criterionIdt(self.G_UM(real_mnist), real_mnist)
        #     loss_g_idt += 0.5 * self.criterionIdt(self.G_UM(real_mnist_rot), real_mnist_rot)
        #     loss += loss_g_idt

        # if self.use_distance_loss:
        #     ####

        loss.backward()
        self.G_optim.step()

        self.loss_G = loss.data.cpu()

    def get_test_visuals(self):
        with torch.no_grad():
            fig, cols = plt.subplots(1, 3)
            for i in range(3):
                try:
                    usps_inputs = next(self.usps_test_iter)[0].cuda()
                except StopIteration:
                    # Reached end of batch, restart iterator
                    self.usps_test_iter = iter(self.usps_test_loader)
                    usps_inputs = next(self.usps_test_iter)[0].cuda()
                mnist_outputs = self.G_UM(usps_inputs)
                # left column: original images (usps inputs)
                inputs_joined = usps_inputs.squeeze().view(-1, self.image_size)
                # right column: transformed images (mnist-ish outputs)
                outputs_joined = mnist_outputs.squeeze().view(-1, self.image_size)
                whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1).cpu()
                cols[i].imshow(whole_grid, cmap='gray')
            fig.show()