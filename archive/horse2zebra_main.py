import argparse
import horse2zebra_datasets
from horse2zebra_plainGAN_solver import Solver

def str2bool(v):
    return v.lower() in 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks')
    parser.add_argument('--which_model_netD', type=str, default='n_layers')
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    # parser.add_argument('--separate_G', type=str2bool, default=True) # Only for GcGAN

    # training hyperparameters
    parser.add_argument('--max_imgs', type=int, default=None)
    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--niter_decay', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr_policy', type=str, default='lambda')

    # loss function/full objective config
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--use_lsgan', type=str2bool, default=True)
    # parser.add_argument('--lambda_cycle', type=float, default=0.0)
    # parser.add_argument('--lambda_gc', type=float, default=0.0)
    # parser.add_argument('--lambda_reconst', type=float, default=0.0)
    # parser.add_argument('--lambda_dist', type=float, default=0.0)
    # parser.add_argument('--lambda_distance_A', type=float, default=0.05)
    # parser.add_argument('--lambda_distance_B', type=float, default=0.1)
    # parser.add_argument('--use_self_distance', required=False, type=str2bool)
    # parser.add_argument('--epoch_count', type=int, default=0) # purpose?
    # parser.add_argument('--unnormalized_distances', required=False, type=str2bool)

    # misc
    parser.add_argument('--begin_train', type=str2bool, default=True)
    # parser.add_argument('--geometry', type=int, default=0) # 0:identity, 1:rot90, 2:rot180, 3:vf, 4:gauss_noise, 5:rot90+noise, 6:rot180+noise
    # parser.add_argument('--noise_var', type=float, default=0.1)
    # parser.add_argument('--pretrained_mnist_model', type=str, default="models/MNISTClassifier/200115-172045-MNISTClassifier.pth")

    config = parser.parse_args()

    horse_loader, zebra_loader = horse2zebra_datasets.get_loaders(config)
    solver = Solver(config, horse_loader, zebra_loader)
    # solver.train()