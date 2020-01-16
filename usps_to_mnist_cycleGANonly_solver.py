from usps_to_mnist_base_solver import AbstractSolver
import torch
import torch.optim as optim
import networks
import GANLosses
import itertools

"""
USPS -> MNIST GcGAN only solver
"""


class Solver(AbstractSolver):
    def __init__(self, config, usps_train_loader, mnist_train_loader, usps_test_loader):
        super().__init__(config, usps_train_loader, mnist_train_loader, usps_test_loader)
        self.criterionCycle = GANLosses.CycleLoss()

    def init_models(self):
        """ Models: G_UM, G_MU, D_M, D_U """
        # Networks
        self.G_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        self.G_MU = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        self.D_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim,
                                     which_model_netD=self.config.which_model_netD,
                                     n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                     gpu_ids=self.gpu_ids)
        self.D_U = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim,
                                     which_model_netD=self.config.which_model_netD,
                                     n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                     gpu_ids=self.gpu_ids)

        # Optimisers
        # single optimiser for both generators
        self.G_optim = optim.Adam(itertools.chain(self.G_UM.parameters(), self.G_MU.parameters()),
                                  self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.D_M_optim = optim.Adam(self.D_M.parameters(),
                                    lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.D_U_optim = optim.Adam(self.D_U.parameters(),
                                    lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.optimizers = [self.G_optim, self.D_M_optim, self.D_U_optim]

        # Schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, self.config))

    def train(self):
        print('----------- USPS->MNIST: Training model -----------')
        n_iters = self.config.niter + self.config.niter_decay
        iter_count = 0
        while True:
            usps_train_iter = iter(self.usps_train_loader)
            mnist_train_iter = iter(self.mnist_train_loader)
            for usps_batch, mnist_batch in zip(usps_train_iter, mnist_train_iter):
                real_usps, u_labels = usps_batch
                real_mnist, m_labels = mnist_batch
                real_usps = real_usps.cuda()
                real_mnist = real_mnist.cuda()

                # Generate
                fake_mnist = self.G_UM.forward(real_usps)
                fake_usps  = self.G_MU.forward(real_mnist)

                # what do D_M and D_U think?
                pred_d_m_fake = self.D_M(fake_mnist)
                pred_d_m_real = self.D_M(real_mnist)
                pred_d_u_fake = self.D_U(fake_usps)
                pred_d_u_real = self.D_U(real_usps)

                # if (iter_count + 1) % 2 == 0:
                # backward D and D_gc. Use a single loss function since D_optim has params of both
                self.backward_D(pred_d_m_fake, pred_d_m_real, pred_d_u_fake, pred_d_u_real)

                """ ----------------------------------------- TO DO ---------------------------------------------"""

                if (iter_count + 1) % 2 == 0:
                # backward G (and hence G_gc at the same time) (use the same batch as above)
                    self.backward_G(real_usps, real_mnist, fake_usps, fake_mnist, pred_d_m_fake, pred_d_u_fake)

                # update learning rates
                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 10 == 0:
                    print("{:04d} of {:04d} iterations. loss_D_M = {:.5f}, loss_D_U = {:.5f}, loss_G = {:.5f}".format(
                        iter_count + 1, n_iters, self.loss_D_M, self.loss_D_U, self.loss_G))
                    self.get_test_visuals()

                iter_count += 1
                # if all iterations done, break out of both loops
                if iter_count >= n_iters:
                    break
            if iter_count >= n_iters:
                break

        # DONE
        print('----------- USPS->MNIST: Finished training -----------')

    def backward_D(self, pred_d_m_fake, pred_d_m_real, pred_d_u_fake, pred_d_u_real):
        self.D_M_optim.zero_grad()
        loss_d_m = self.criterionGAN(pred_d_m_fake.cpu(), False) + self.criterionGAN(pred_d_m_real.cpu(), True)
        loss_d_m = loss_d_m.cuda()
        loss_d_m.backward(retain_graph=True)
        self.D_M_optim.step()
        self.loss_D_M = loss_d_m.cpu()

        self.D_U_optim.zero_grad()
        loss_d_u = self.criterionGAN(pred_d_u_fake.cpu(), False) + self.criterionGAN(pred_d_u_real.cpu(), True)
        loss_d_u = loss_d_u.cuda()
        loss_d_u.backward(retain_graph=True)
        self.D_U_optim.step()
        self.loss_D_U = loss_d_u.cpu()

    def backward_G(self, real_usps, real_mnist, fake_usps, fake_mnist, pred_d_m_fake, pred_d_u_fake):
        self.G_optim.zero_grad()

        # GAN loss
        loss_g_gan = self.criterionGAN(pred_d_m_fake.cpu(), True) * 0.5
        loss_g_gan += self.criterionGAN(pred_d_u_fake.cpu(), True) * 0.5
        loss_g_gan *= self.config.lambda_gan
        loss = loss_g_gan.cuda()

        # Cycle consistency loss
        loss_g_gc = self.criterionCycle(real_usps, real_mnist, fake_usps, fake_mnist, self.G_UM, self.G_MU)
        loss_g_gc *= self.config.lambda_gc
        loss += loss_g_gc.cuda()

        # Reconstruction loss: CycleGAN version
        if self.config.lambda_reconst > 0:
            loss_g_idt = 0.5 * self.criterionReconst(self.G_UM(real_mnist), real_mnist)
            loss_g_idt += 0.5 * self.criterionReconst(self.G_MU(real_usps), real_usps)
            loss_g_idt *= self.config.lambda_reconst
            loss += loss_g_idt.cuda()

        # if self.config.lambda_dist > 0:
        #     ####
        #   ... loss_g_dist *= self.config.lambda_dist

        loss.backward()
        self.G_optim.step()

        self.loss_G = loss.data.cpu()
