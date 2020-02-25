from horse2zebra_base_solver import AbstractImg2ImgSolver
import networks
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()


class Solver(AbstractImg2ImgSolver):
    def __init__(self, config, A_loader, B_loader):
        super().__init__(config, A_loader, B_loader)

    def init_models(self):
        # Networks
        self.G_AB = networks.define_G(input_nc=3, output_nc=3, ngf=self.config.g_conv_dim,
                                      which_model_netG=self.config.which_model_netG, norm='batch', init_type='normal',
                                      gpu_ids=self.gpu_ids)
        self.D_B = networks.define_D(input_nc=3, ndf=self.config.d_conv_dim, which_model_netD=self.config.which_model_netD,
                                         n_layers_D=6, norm='instance', use_sigmoid=True, init_type='normal',
                                         gpu_ids=self.gpu_ids)
        # self.G_AB = horse2zebra_networks.G12(self.config, self.config.g_conv_dim).cuda()
        # self.D_B = horse2zebra_networks.D2(self.config.d_conv_dim).cuda()

        # Optimisers
        self.G_optim = optim.Adam(self.G_AB.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.D_optim = optim.Adam(self.D_B.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
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
            for real_A, real_B in zip(A_dataiter, B_dataiter):
                real_A = cuda(real_A)
                real_B = cuda(real_B)
                A_processed += len(real_A)
                B_processed += len(real_B)

                # Generate
                fake_B = self.G_AB(real_A)
                pred_D_B_real = self.D_B(real_B)
                pred_D_B_fake = self.D_B(fake_B)

                # Discriminator accuracy on fake & real B
                with torch.no_grad():
                    fake_B_guesses = pred_D_B_fake.squeeze().cpu().numpy().round()
                    D_B_correct_fake += np.sum(fake_B_guesses == self.target_fake_label)
                    real_mnist_guesses = pred_D_B_real.squeeze().cpu().numpy().round()
                    D_B_correct_real += np.sum(real_mnist_guesses == self.target_real_label)

                # Calculate losses and backpropagate
                loss_D_sum += self.get_D_loss_and_bkwd(pred_D_B_real, pred_D_B_fake)
                loss_G_sum += self.get_G_loss_and_bkwd(real_A, real_B, fake_B, pred_D_B_fake)

                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 8 == 0:
                    loss_D_avg = loss_D_sum / 8
                    loss_G_avg = loss_G_sum / 8
                    D_B_accuracy_real = D_B_correct_real / A_processed * 100
                    D_B_accuracy_fake = D_B_correct_fake / B_processed * 100
                    self.report_results(iter_count, n_iters, loss_D_avg, loss_G_avg, D_B_accuracy_real,
                                        D_B_accuracy_fake, real_A, fake_B)
                    print(f"loss_D_avg = {loss_D_avg}, loss_G_avg = {loss_G_avg}")
                    loss_D_sum = 0
                    loss_G_sum = 0

                    # show before and after
                    fig = plt.figure(figsize=(6, 6))
                    gs = fig.add_gridspec(1, 2)
                    ax0 = fig.add_subplot(gs[0, 0])
                    ax1 = fig.add_subplot(gs[0, 1])
                    real_A = real_A[0].cpu().detach().permute(1, 2, 0)
                    ax0.imshow(real_A)
                    fake_B = fake_B[0].cpu().detach().permute(1, 2, 0)
                    ax1.imshow(fake_B)
                    fig.show()

                iter_count += 1
                if iter_count >= n_iters:
                    break
            if iter_count >= n_iters:
                break

    def get_D_loss_and_bkwd(self, pred_D_B_real, pred_D_B_fake):
        self.D_optim.zero_grad()

        # D tries to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss = self.criterionGAN(pred_D_B_real.cpu(), True) + self.criterionGAN(pred_D_B_fake.cpu(), False)
        loss = cuda(loss)
        loss.backward(retain_graph=True)
        self.D_optim.step()

        return loss.data.cpu()

    def get_G_loss_and_bkwd(self, real_A, real_B, fake_B, pred_D_B_fake):
        self.G_optim.zero_grad()

        loss = self.criterionGAN(pred_D_B_fake.cpu(), True)
        loss = cuda(loss)
        loss.backward()
        self.G_optim.step()

        return loss.data.cpu()