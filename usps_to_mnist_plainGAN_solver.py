from usps_to_mnist_base_solver import AbstractSolver
import torch.optim as optim
import networks

"""
USPS -> MNIST plain vanilla GAN solver
"""


class Solver(AbstractSolver):
    def __init__(self, config, usps_train_loader, mnist_train_loader, usps_test_loader):
        super().__init__(config, usps_train_loader, mnist_train_loader, usps_test_loader)

    def init_models(self):
        """ Models: G_UM, D_M """
        # Networks
        self.G_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        self.D_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim, which_model_netD=self.config.which_model_netD,
                                         n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                         gpu_ids=self.gpu_ids)

        # Optimisers
        self.G_optim = optim.Adam(self.G_UM.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2))
        self.D_optim = optim.Adam(self.D_M.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2))
        self.optimizers = [self.G_optim, self.D_optim]

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
                usps, u_labels = usps_batch
                mnist, m_labels = mnist_batch
                usps = usps.cuda()
                mnist = mnist.cuda()

                # Generate
                fake_mnist = self.G_UM.forward(usps)

                # what does D think?
                pred_d_fake = self.D_M(fake_mnist)
                pred_d_real = self.D_M(mnist)

                # if (iter_count + 1) % 2 == 0:
                self.backward_D(pred_d_fake, pred_d_real)

                if (iter_count + 1) % 2 == 0:
                # backward G (and hence G_gc at the same time) (use the same batch as above)
                    self.backward_G(pred_d_fake, mnist)

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
        print('----------- USPS->MNIST: Finished training -----------')

    def backward_D(self, pred_d_fake, pred_d_real):
        self.D_optim.zero_grad()

        # D trying to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss = (self.criterionGAN(pred_d_fake.cpu(), False) + self.criterionGAN(pred_d_real.cpu(), True))
        loss = loss.cuda()
        loss.backward(retain_graph=True)
        self.D_optim.step()

        self.loss_D = loss.cpu()

    def backward_G(self, pred_d_fake, real_mnist):
        self.G_optim.zero_grad()

        loss = self.criterionGAN(pred_d_fake.cpu(), True)

        # Reconstruction loss: Vanilla GAN version
        if self.config.lambda_reconst > 0:
            loss_g_idt = self.criterionReconst(self.G_UM(real_mnist), real_mnist)
            loss_g_idt *= self.config.lambda_reconst
            loss += loss_g_idt

        loss = loss.cuda()
        loss.backward()
        self.G_optim.step()

        self.loss_G = loss.data.cpu()
