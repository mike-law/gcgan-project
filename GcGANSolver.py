from AbstractImg2ImgSolver import AbstractImg2ImgSolver
import networks
import torch
import torch.optim as optim
import itertools
import GANLosses
import numpy as np


def cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()


class Solver(AbstractImg2ImgSolver):
    def __init__(self, config, A_loader, B_loader):
        super().__init__(config, A_loader, B_loader)
        self.f, self.f_inv = self.get_geo_transform(self.config.geometry)
        self.criterionGc = GANLosses.GCLoss() # idea: for combining multiple loss criteria, have an array of loss objs

    def init_models(self):
        # Networks
        self.G_AB = networks.define_G(input_nc=self.config.input_nc, output_nc=self.config.output_nc, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch',
                                      init_type='normal', gpu_ids=self.gpu_ids)
        if self.config.separate_G:
            self.G_gc_AB = networks.define_G(input_nc=self.config.input_nc, output_nc=self.config.output_nc, ngf=self.config.g_conv_dim,
                                             which_model_netG=self.config.which_model_netG, norm='batch',
                                             init_type='normal', gpu_ids=self.gpu_ids)
        self.D_B = networks.define_D(input_nc=self.config.input_nc, ndf=self.config.d_conv_dim, which_model_netD=self.config.which_model_netD,
                                    n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                    gpu_ids=self.gpu_ids, image_size=self.config.image_size)
        self.D_gc_B = networks.define_D(input_nc=self.config.input_nc, ndf=self.config.d_conv_dim, which_model_netD=self.config.which_model_netD,
                                        n_layers_D=3, norm='instance', use_sigmoid=True, init_type='normal',
                                        gpu_ids=self.gpu_ids, image_size=self.config.image_size)

        # Optimisers
        if self.config.separate_G:
            self.G_optim = optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_gc_AB.parameters()),
                                      lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        else:
            self.G_optim = optim.Adam(self.G_AB.parameters(),
                                      lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.D_optim = optim.Adam(itertools.chain(self.D_B.parameters(), self.D_gc_B.parameters()),
                                  lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.optimizers = [self.G_optim, self.D_optim]

        # Schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, self.config))

    def train(self):
        n_iters = self.config.niter + self.config.niter_decay
        iter_count = 0
        loss_D_sum = 0
        loss_G_sum = 0
        D_B_correct_fake = 0
        D_B_correct_real = 0
        A_processed = 0
        B_processed = 0

        while True:
            A_dataiter = iter(self.A_loader)
            B_dataiter = iter(self.B_loader)
            for a, b in zip(A_dataiter, B_dataiter):
                if self.config.has_labels:
                    a = cuda(a[0]) # only take the image from (image, label) tuple
                    b = cuda(b[0])
                else:
                    a = cuda(a)
                    b = cuda(b)
                a = cuda(a)
                b = cuda(b)
                f_a = self.f(a)
                f_b = self.f(b)
                A_processed += len(a)
                B_processed += len(b)

                # Generate
                G_a = self.G_AB(a)
                if self.config.separate_G:
                    G_f_a = self.G_gc_AB(f_a)
                else:
                    G_f_a = self.G_AB(f_a)

                # Discriminator predictions
                pred_D_B_real = self.D_B(b)
                pred_D_gc_B_real = self.D_gc_B(f_b)
                pred_D_B_fake = self.D_B(G_a)
                pred_D_gc_B_fake = self.D_gc_B(G_f_a)

                # Discriminator accuracy on fake & real B
                with torch.no_grad():
                    fake_B_guesses = pred_D_B_fake.squeeze().cpu().numpy().round()
                    D_B_correct_fake += np.sum(fake_B_guesses == self.target_fake_label)
                    real_mnist_guesses = pred_D_B_real.squeeze().cpu().numpy().round()
                    D_B_correct_real += np.sum(real_mnist_guesses == self.target_real_label)

                # Calculate losses and backpropagate
                loss_D_sum += self.get_D_loss_and_bkwd(pred_D_B_real, pred_D_gc_B_real, pred_D_B_fake, pred_D_gc_B_fake)
                loss_G_sum += self.get_G_loss_and_bkwd(G_a, G_f_a, pred_D_B_fake, pred_D_gc_B_fake)

                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 10 == 0:
                    loss_D_avg = loss_D_sum / 10
                    loss_G_avg = loss_G_sum / 10
                    D_B_accuracy_real = D_B_correct_real / A_processed * 100
                    D_B_accuracy_fake = D_B_correct_fake / B_processed * 100
                    self.report_results(iter_count, n_iters, loss_D_avg, loss_G_avg, D_B_accuracy_real,
                                        D_B_accuracy_fake, a, G_a)
                    loss_D_sum = 0
                    loss_G_sum = 0
                    A_processed = 0
                    B_processed = 0
                    D_B_correct_real = 0
                    D_B_correct_fake = 0

                iter_count += 1
                if iter_count >= n_iters:
                    break
            if iter_count >= n_iters:
                break

    def get_D_loss_and_bkwd(self, pred_D_B_real, pred_D_gc_B_real, pred_D_B_fake, pred_D_gc_B_fake):
        self.D_optim.zero_grad()

        # D tries to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss = self.criterionGAN(pred_D_B_real.cpu(), True) + self.criterionGAN(pred_D_B_fake.cpu(), False)
        loss += self.criterionGAN(pred_D_gc_B_real.cpu(), True) + self.criterionGAN(pred_D_gc_B_fake.cpu(), False)
        loss = cuda(0.5 * loss)
        loss.backward(retain_graph=True)
        self.D_optim.step()

        return loss.data.cpu()

    def get_G_loss_and_bkwd(self, G_a, G_f_a, pred_D_B_fake, pred_D_gc_B_fake):
        self.G_optim.zero_grad()

        # GAN loss
        loss_gan = self.criterionGAN(pred_D_B_fake.cpu(), True)
        loss_gan += self.criterionGAN(pred_D_gc_B_fake.cpu(), True)
        loss_gan = cuda(0.5 * loss_gan)

        # GC loss
        loss_gc = self.criterionGc(G_a, G_f_a, self.f, self.f_inv)
        loss_gc = cuda(loss_gc)

        loss = loss_gan + loss_gc
        loss.backward()
        self.G_optim.step()

        return loss.data.cpu()

    # Geometric transformations
    def rot90(self, tensor, direction):  # 0 = clockwise, 1 = counterclockwise
        t = torch.transpose(tensor, 2, 3).cuda()
        inv_idx = torch.arange(self.config.image_size - 1, -1, -1).long().cuda()
        if direction == 0:
            t = torch.index_select(t, 3, inv_idx)
        elif direction == 1:
            t = torch.index_select(t, 2, inv_idx)
        return t

    def rot180(self, tensor):
        inv_idx = torch.arange(self.config.image_size - 1, -1, -1).long().cuda()
        tensor = torch.index_select(tensor, 2, inv_idx)
        tensor = torch.index_select(tensor, 3, inv_idx)
        return tensor

    def vf(self, tensor):
        inv_idx = torch.arange(self.config.image_size - 1, -1, -1).long().cuda()
        return torch.index_select(tensor, 2, inv_idx)

    def noiser(self, tensor):
        noise = torch.randn_like(tensor)
        noise = noise * np.sqrt(self.config.noise_var)
        noise = noise.cuda()
        return tensor + noise

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
