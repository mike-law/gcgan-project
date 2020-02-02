import argparse
import usps_mnist_datasets_16x16, usps_mnist_datasets_28x28
import os
from datetime import datetime


def str2bool(v):
    return v.lower() in 'true'


def save(solver):
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    new_folder_path = "./testruns/" + timestamp
    os.mkdir(new_folder_path)
    solver.save_testrun(new_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--which_model_netG', type=str, default='resnet_6blocks')
    parser.add_argument('--which_model_netD', type=str, default='basic')
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--separate_G', type=str2bool, default=True) # Only for GcGAN
    # parser.add_argument('--num_classes', type=int, default=10)

    # training hyperparameters
    parser.add_argument('--max_imgs_per_digit', type=int, default=500)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--niter_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr_policy', type=str, default='lambda')

    # loss function/full objective config
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--use_lsgan', type=str2bool, default=True)
    parser.add_argument('--lambda_cycle', type=float, default=0.0)
    parser.add_argument('--lambda_gc', type=float, default=0.0)
    parser.add_argument('--lambda_reconst', type=float, default=0.0)
    parser.add_argument('--lambda_dist', type=float, default=0.0)
    # parser.add_argument('--lambda_distance_A', type=float, default=0.05)
    # parser.add_argument('--lambda_distance_B', type=float, default=0.1)
    # parser.add_argument('--use_self_distance', required=False, type=str2bool)
    # parser.add_argument('--epoch_count', type=int, default=0) # purpose?
    # parser.add_argument('--unnormalized_distances', required=False, type=str2bool)

    # misc
    parser.add_argument('--begin_train', type=str2bool, default=True)
    parser.add_argument('--geometry', type=int, default=0) # 0:identity, 1:rot90, 2:rot180, 3:vf, 4:gauss_noise, 5:rot90+noise, 6:rot180+noise
    parser.add_argument('--noise_var', type=float, default=0.1)
    parser.add_argument('--pretrained_mnist_model', type=str, default="models/MNISTClassifier/200115-172045-MNISTClassifier.pth")

    config = parser.parse_args()

    print("USPS->MNIST: Getting train and test loaders...", end=' ')
    usps_train_loader, mnist_train_loader = None, None
    if config.image_size == 16:
        usps_train_loader, mnist_train_loader, usps_test_loader =\
            usps_mnist_datasets_16x16.get_loaders(config)
    elif config.image_size == 28:
        usps_train_loader, mnist_train_loader = usps_mnist_datasets_28x28.get_loaders(config)
    print("done")

    if config.lambda_gc == 0 and config.lambda_cycle == 0:
        from usps_to_mnist_plainGAN_solver import Solver
    elif config.lambda_gc > 0 and config.lambda_cycle == 0:
        from usps_to_mnist_gcGANonly_solver import Solver
    elif config.lambda_gc == 0 and config.lambda_cycle > 0:
        from usps_to_mnist_cycleGANonly_solver import Solver
    elif config.lambda_gc > 0 and config.lambda_cycle > 0:
        from usps_to_mnist_cycleGcGAN_solver import Solver

    solver = Solver(config, usps_train_loader, mnist_train_loader)  # usps_test_loader)

    if config.begin_train:
        solver.train()

        print("----------- Begin testing stage -----------")
        # fake_mnist_loader = usps_mnist_datasets_28x28.get_fake_mnist_loader(solver)
        # solver.test(fake_mnist_loader)
        save(solver)

        # if input('view images? y/n: ').lower() == 'y':
        #     while True:
        #         solver.get_test_visuals()
        #         if input('more images? y/n: ').lower() == 'n':
        #             break
