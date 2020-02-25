from usps_to_mnist_base_solver import AbstractSolver
import torch
import torch.optim as optim
import networks
import itertools
import GANLosses
import numpy as np

"""
USPS -> MNIST GcGAN only solver
"""


class Solver(AbstractSolver):
    def __init__(self, config, usps_train_loader, mnist_train_loader):
        super().__init__(config, usps_train_loader, mnist_train_loader)
        self.f, self.f_inv = self.get_geo_transform(self.config.geometry)
        self.criterionGc = GANLosses.GCLoss()

    def init_models(self):
        """ Models: G_UM (G_gc_UM if separate_G is True), D_M, D_gc_M """
        # Networks
        self.G_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        if self.config.separate_G:
            self.G_gc_UM = networks.define_G(input_nc=1, output_nc=1, ngf=self.config.g_conv_dim,
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
        if self.config.separate_G:
            self.G_optim = optim.Adam(itertools.chain(self.G_UM.parameters(), self.G_gc_UM.parameters()),
                                      lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        else:
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
        # Stats
        loss_D_sum = 0
        loss_G_sum = 0
        d_correct_real = 0
        d_correct_fake = 0
        correctly_labelled = 0
        usps_processed = 0
        mnist_processed = 0
        while True:
            usps_train_iter = iter(self.usps_train_loader)
            mnist_train_iter = iter(self.mnist_train_loader)
            for usps_batch, mnist_batch in zip(usps_train_iter, mnist_train_iter):
                real_usps, u_labels = usps_batch
                real_mnist, m_labels = mnist_batch
                real_usps = real_usps.cuda()
                real_mnist = real_mnist.cuda()
                usps_processed += len(u_labels)
                mnist_processed += len(m_labels)

                # Generate
                f_mnist = self.f(real_mnist)
                fake_mnist = self.G_UM.forward(real_usps)
                if self.config.separate_G:
                    f_fake_mnist = self.G_gc_UM.forward(self.f(real_usps))
                else:
                    f_fake_mnist = self.G_UM.forward(self.f(real_usps))
                pred_d_fake = self.D_M(fake_mnist)
                pred_d_real = self.D_M(real_mnist)
                pred_d_gc_fake = self.D_gc_M(f_fake_mnist)
                pred_d_gc_real = self.D_gc_M(f_mnist)

                # Classification accuracy on fake MNIST images
                correctly_labelled += self.get_train_accuracy(fake_mnist, u_labels)

                # Discriminator accuracy on fake & real MNIST
                with torch.no_grad():
                    fake_mnist_guesses = pred_d_fake.squeeze().cpu().numpy().round()
                    d_correct_fake += np.sum(fake_mnist_guesses == self.target_fake_label)
                    real_mnist_guesses = pred_d_real.squeeze().cpu().numpy().round()
                    d_correct_real += np.sum(real_mnist_guesses == self.target_real_label)

                # Calculate losses and backpropagate
                loss_D_sum += self.backward_D(pred_d_fake, pred_d_gc_fake, pred_d_real, pred_d_gc_real)
                loss_G_sum += self.backward_G(real_usps, fake_mnist, f_fake_mnist, real_mnist, f_mnist, pred_d_fake,
                                              pred_d_gc_fake)

                # update learning rates
                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 10 == 0:
                    loss_D_avg = loss_D_sum / 10
                    loss_G_avg = loss_G_sum / 10
                    fake_mnist_class_acc = correctly_labelled / usps_processed * 100
                    d_acc_real_mnist = d_correct_real / mnist_processed * 100
                    d_acc_fake_mnist = d_correct_fake / usps_processed * 100
                    self.report_results(iter_count, n_iters, loss_D_avg, loss_G_avg, fake_mnist_class_acc,
                                        d_acc_real_mnist, d_acc_fake_mnist, real_usps, fake_mnist)
                    loss_D_sum, loss_G_sum = 0, 0
                    d_correct_real, d_correct_fake = 0, 0
                    correctly_labelled, usps_processed, mnist_processed = 0, 0, 0

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

        return loss.data.cpu()

    def backward_G(self, real_usps, fake_mnist, f_fake_mnist, real_mnist, f_real_mnist, pred_d_fake, pred_d_gc_fake):
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
            loss_g_reconst = 0.5 * self.criterionReconst(self.G_UM(real_mnist), real_mnist)
            if self.config.separate_G:
                loss_g_reconst += 0.5 * self.criterionReconst(self.G_gc_UM(f_real_mnist), f_real_mnist)
            else:
                loss_g_reconst += 0.5 * self.criterionReconst(self.G_UM(f_real_mnist), f_real_mnist)
            loss_g_reconst *= self.config.lambda_reconst
            loss += loss_g_reconst.cuda()

        # if self.config.lambda_dist > 0:
        #     ####
        #   ... loss_g_dist *= self.config.lambda_dist

        loss.backward()
        self.G_optim.step()

        return loss.data.cpu()

    def rot90(self, tensor, direction): # 0 = clockwise, 1 = counterclockwise
        t = torch.transpose(tensor, 2, 3).cuda()
        inv_idx = torch.arange(self.config.image_size-1, -1, -1).long().cuda()
        if direction == 0:
            t = torch.index_select(t, 3, inv_idx)
        elif direction == 1:
            t = torch.index_select(t, 2, inv_idx)
        return t

    def rot180(self, tensor):
        inv_idx = torch.arange(self.config.image_size-1, -1, -1).long().cuda()
        tensor = torch.index_select(tensor, 2, inv_idx)
        tensor = torch.index_select(tensor, 3, inv_idx)
        return tensor

    def vf(self, tensor):
        inv_idx = torch.arange(self.config.image_size-1, -1, -1).long().cuda()
        return torch.index_select(tensor, 2, inv_idx)

    def noiser(self, img_batch):
        noise = torch.randn_like(img_batch)
        noise = noise * np.sqrt(self.config.noise_var)
        noise = noise.cuda()
        return img_batch + noise

    def get_geo_transform(self, transform_id):
        """ Returns f and f^-1 """
        if transform_id == 0:
            idt = lambda img: img
            return idt, idt
        if transform_id == 1:
            clockwise = lambda img: self.rot90(img, 0)
            anticlockwise = lambda img: self.rot90(img, 1)
            return clockwise, anticlockwise
        elif transform_id == 2:
            return self.rot180, self.rot180
        elif transform_id == 3:
            return self.vf, self.vf
        elif transform_id == 4:
            idt = lambda img: img
            return self.noiser, idt  # possible to make it so that f^-1 removes the noise added from noiser?
        elif transform_id == 5:
            transf = lambda img: self.noiser(self.rot90(img, 0))
            inv = lambda img: self.rot90(img, 1)
            return transf, inv
        elif transform_id == 6:
            transf = lambda img: self.noiser(self.rot180(img))
            return transf, self.rot180
