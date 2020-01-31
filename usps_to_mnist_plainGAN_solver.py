from usps_to_mnist_base_solver import AbstractSolver
import torch
import torch.optim as optim
import networks
import numpy as np
import mnist_model

"""
USPS -> MNIST plain vanilla GAN solver
"""


class Solver(AbstractSolver):
    def __init__(self, config, usps_train_loader, mnist_train_loader):
        super().__init__(config, usps_train_loader, mnist_train_loader)

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
        # Stats
        loss_D_sum = 0
        loss_G_sum = 0
        d_correct_real = 0  # % of real MNIST the discriminator correctly identifies as real
        d_correct_fake = 0  # % of fake MNIST the discriminator correctly identifies as fake
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
                fake_mnist = self.G_UM.forward(real_usps)
                pred_d_fake = self.D_M(fake_mnist)
                pred_d_real = self.D_M(real_mnist)

                # Classification accuracy on fake MNIST images
                correctly_labelled += self.get_train_accuracy(fake_mnist, u_labels)

                # Discriminator accuracy on fake & real MNIST
                with torch.no_grad():
                    fake_mnist_guesses = pred_d_fake.squeeze().cpu().numpy().round()
                    d_correct_fake += np.sum(fake_mnist_guesses == self.target_fake_label)
                    real_mnist_guesses = pred_d_real.squeeze().cpu().numpy().round()
                    d_correct_real += np.sum(real_mnist_guesses == self.target_real_label)

                # Calculate losses and backpropagate
                loss_D_sum += self.backward_D(pred_d_fake, pred_d_real)
                loss_G_sum += self.backward_G(real_usps, pred_d_fake, real_mnist, fake_mnist)

                """ Trying to set up the image buffer causes CUDA memory error. Why???? """
                # if iter_count > 0:
                #     # For fake MNIST images, take half of fake_mnist, and half of fake_mnist_buffer
                #     with torch.no_grad():
                #         indices = torch.randperm(self.config.batch_size)
                #         indices = indices[:(self.config.batch_size//2)]
                #         fake_mnist_subset = fake_mnist[indices]
                #         fake_mnist_buffer_subset = self.fake_mnist_buffer[indices] # use same set of indices
                #         input_to_D = torch.cat((fake_mnist_subset, fake_mnist_buffer_subset))
                #     pred_d_fake = self.D_M(input_to_D)
                #     pred_d_real = self.D_M(real_mnist)
                #     self.backward_D(pred_d_fake, pred_d_real)
                #     self.backward_G(pred_d_fake, real_mnist)
                #
                #     # select half the imgs from fake_mnist to replace half the images in fake_mnist_buffer
                #     indices = torch.randperm(self.config.batch_size)
                #     indices = indices[:self.config.batch_size//2]
                #     for idx in indices:
                #         self.fake_mnist_buffer[idx] = fake_mnist[idx].clone()
                # else:
                #     # set up buffer. Just place all in so the buffer initialises with the right size
                #     self.fake_mnist_buffer = fake_mnist.clone()

                # update learning rates

                for sched in self.schedulers:
                    sched.step()

                if (iter_count + 1) % 10 == 0:
                    # Display statistics averaged across the last 10 iterations
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

    def backward_D(self, pred_d_fake, pred_d_real):
        self.D_optim.zero_grad()

        # D trying to maximise the probability that it is right. So tries to minimise the prob of wrong
        loss = (self.criterionGAN(pred_d_fake.cpu(), False) + self.criterionGAN(pred_d_real.cpu(), True))
        loss = loss.cuda()
        loss.backward(retain_graph=True)
        self.D_optim.step()

        return loss.data.cpu()

    def backward_G(self, real_usps, pred_d_fake, real_mnist, fake_mnist):
        self.G_optim.zero_grad()

        loss = self.criterionGAN(pred_d_fake.cpu(), True).cuda()

        # Reconstruction loss: Vanilla GAN version
        if self.config.lambda_reconst > 0:
            loss_g_reconst = self.criterionReconst(self.G_UM(real_mnist), real_mnist)
            loss_g_reconst *= self.config.lambda_reconst
            loss += loss_g_reconst.cuda()

        # if self.config.lambda_dist > 0:
        #     # Calculate means and standard deviations of pairwise distances within real minibatches
        #     loss_g_dist = self.criterionDist(real_usps, fake_mnist, self.mean_dist_usps, self.mean_dist_mnist,
        #                                      self.stdev_dist_usps, self.stdev_dist_mnist, self.config.batch_size)
        #     loss_g_dist *= self.config.lambda_dist
        #     loss += loss_g_dist.cuda()

        loss = loss.cuda()
        loss.backward()
        self.G_optim.step()

        return loss.data.cpu()