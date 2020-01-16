from usps_to_mnist_base_solver import AbstractSolver
import torch
import torch.optim as optim
import networks
import itertools
import GANLosses

"""
USPS -> MNIST GcGAN only solver
"""


class Solver(AbstractSolver):
    def __init__(self, config, usps_train_loader, mnist_train_loader, usps_test_loader):
        super().__init__(config, usps_train_loader, mnist_train_loader, usps_test_loader)
        self.f, self.f_inv = self.get_geo_transform(self.config.geometry)
        self.criterionGc = GANLosses.GCLoss()

    def init_models(self):
        """ Models: G_UM, D_M, D_gc_M """
        # Networks
        self.G_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        self.D_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim,
                                     which_model_netD=self.config.which_model_netD,
                                     n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                     gpu_ids=self.gpu_ids)
        self.D_gc_M = networks.define_D(input_nc=1, ndf=self.config.d_conv_dim,
                                       which_model_netD=self.config.which_model_netD,
                                       n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                       gpu_ids=self.gpu_ids)

        # Optimisers
        self.G_optim = optim.Adam(self.G_UM.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2))
        self.D_optim = optim.Adam(itertools.chain(self.D_M.parameters(), self.D_gc_M.parameters()),
                                  lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
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
                f_mnist = self.f(mnist)
                fake_mnist = self.G_UM.forward(usps)
                f_fake_mnist = self.G_UM.forward(self.f(usps))

                # what do D and D_gc think?
                pred_d_fake = self.D_M(fake_mnist)
                pred_d_real = self.D_M(mnist)
                pred_d_gc_fake = self.D_gc_M(f_fake_mnist)
                pred_d_gc_real = self.D_gc_M(f_mnist)

                # if (iter_count + 1) % 2 == 0:
                # backward D and D_gc. Use a single loss function since D_optim has params of both
                self.backward_D(pred_d_fake, pred_d_gc_fake, pred_d_real, pred_d_gc_real)

                # if (iter_count + 1) % 2 == 0:
                # backward G (and hence G_gc at the same time) (use the same batch as above)
                self.backward_G(fake_mnist, f_fake_mnist, mnist, f_mnist, pred_d_fake, pred_d_gc_fake)

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

    def backward_D(self, pred_d_fake, pred_d_gc_fake, pred_d_real, pred_d_gc_real):
        self.D_optim.zero_grad()

        # D trying to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss_d = self.criterionGAN(pred_d_fake.cpu(), False) + self.criterionGAN(pred_d_real.cpu(), True)
        loss_d_gc = self.criterionGAN(pred_d_gc_fake.cpu(), False) + self.criterionGAN(pred_d_gc_real.cpu(), True)

        loss = 0.5 * (loss_d + loss_d_gc)
        loss = loss.cuda()
        loss.backward(retain_graph=True)
        self.D_optim.step()

        self.loss_D = loss.cpu()

    def backward_G(self, fake_mnist, f_fake_mnist, real_mnist, f_real_mnist, pred_d_fake, pred_d_gc_fake):
        self.G_optim.zero_grad()

        # GAN loss
        loss_g_gan = self.criterionGAN(pred_d_fake.cpu(), True) * 0.5
        loss_g_gan += self.criterionGAN(pred_d_gc_fake.cpu(), True) * 0.5
        loss_g_gan *= self.config.lambda_gan
        loss = loss_g_gan.cuda()

        # GC loss
        loss_g_gc = self.criterionGc(fake_mnist, f_fake_mnist, self.f, self.f_inv)
        loss_g_gc *= self.config.lambda_gc
        loss += loss_g_gc.cuda()

        # Reconstruction loss: GcGAN version
        if self.config.lambda_reconst > 0:
            loss_g_idt = 0.5 * self.criterionReconst(self.G_UM(real_mnist), real_mnist)
            loss_g_idt += 0.5 * self.criterionReconst(self.G_UM(f_real_mnist), f_real_mnist)
            loss_g_idt *= self.config.lambda_reconst
            loss += loss_g_idt.cuda()

        # if self.config.lambda_dist > 0:
        #     ####
        #   ... loss_g_dist *= self.config.lambda_dist

        loss.backward()
        self.G_optim.step()

        self.loss_G = loss.data.cpu()

    def rot90(self, tensor, direction): # 0 = clockwise, 1 = counterclockwise
        t = torch.transpose(tensor, 2, 3).cuda()
        inv_idx = torch.arange(self.config.image_size-1, -1, -1).long().cuda()

        if direction == 0:
            t = torch.index_select(t, 3, inv_idx)
        elif direction == 1:
            t = torch.index_select(t, 2, inv_idx)

        return t

    def vf(self, tensor):
        inv_idx = torch.arange(self.config.image_size-1, -1, -1).long().cuda()
        return torch.index_select(tensor, 2, inv_idx)

    def get_geo_transform(self, transform_id):
        """ Returns f and f^-1 """
        # if transform_id == 0:
        #    return distance GAN??
        if transform_id == 1:
            clockwise = lambda img: self.rot90(img, 0)
            anticlockwise = lambda img: self.rot90(img, 1)
            return clockwise, anticlockwise
        elif transform_id == 2:
            return self.vf, self.vf
