import argparse
import usps_mnist_datasets_16x16, usps_mnist_datasets_28x28
import torch
import matplotlib.pyplot as plt
from usps_to_mnist_solver import Solver


def str2bool(v):
    return v.lower() in 'true'


def get_visuals(slvr, test_iter):
    with torch.no_grad():
        usps_inputs = next(test_iter)[0]
        mnist_outputs = slvr.G_UM(usps_inputs)
        # left column: original images (usps inputs)
        inputs_joined = usps_inputs.squeeze().view(-1, slvr.image_size)
        # right column: transformed images (mnist-ish outputs)
        outputs_joined = mnist_outputs.squeeze().view(-1, slvr.image_size)
        whole_grid = torch.cat((inputs_joined, outputs_joined), dim=1)
        plt.imshow(whole_grid)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--image_size', type=int, default=16)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    # parser.add_argument('--use_reconst_loss', default=True, type=str2bool)
    # parser.add_argument('--use_distance_loss', required=False, type=str2bool)
    # parser.add_argument('--num_classes', type=int, default=10)

    # training hyperparameters
    parser.add_argument('--epoch_count', type=int, default=0)
    parser.add_argument('--niter', type=int, default=64)
    parser.add_argument('--niter_decay', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr_policy', type=str, default='lambda')

    # how many of these are necessary? (for now)
    parser.add_argument('--lambda_distance_A', type=float, default=0.05)
    parser.add_argument('--lambda_distance_B', type=float, default=0.1)
    parser.add_argument('--use_self_distance', required=False, type=str2bool)
    parser.add_argument('--max_items', type=int, default=400)
    parser.add_argument('--unnormalized_distances', required=False, type=str2bool)
    parser.add_argument('--lambda_gc', type=float, default=3.0)

    # 0:distanceGAN, 1:rot, 2:vf
    # parser.add_argument('--geometry', type=int, default=0)
    parser.add_argument('--train', type=str2bool, default=True)

    train_config = parser.parse_args()

    usps_loader, mnist_loader = None, None
    if train_config.image_size == 16:
        usps_loader, mnist_loader = usps_mnist_datasets_16x16.get_train_loaders(train_config)
    elif train_config.image_size == 28:
        usps_loader, mnist_loader = usps_mnist_datasets_28x28.get_train_loaders(train_config)

    solver = Solver(train_config, usps_loader, mnist_loader)

    if input('train? y/n: ').lower() == 'y':
        solver.train()
        if input('view images? y/n: ').lower() == 'y':
            solver.G_UM.cpu()
            if train_config.image_size == 16:
                usps_test_loader = usps_mnist_datasets_16x16.get_usps_test_loader(batch_size=4)
            elif train_config.image_size == 28:
                usps_test_loader = usps_mnist_datasets_28x28.get_usps_test_loader(batch_size=4)
            usps_test_iter = iter(usps_test_loader)
            while True:
                get_visuals(solver, usps_test_iter)
                if input('more images? y/n: ').lower() == 'n':
                    break
