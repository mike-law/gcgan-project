#!/bin/bash
#BATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output out.txt
#SBATCH --error error.txt

# Identity, one G (i.e. Vanilla GAN)
python usps_to_mnist_main.py
# Identity, separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 0 --separate_G 'True'
# Rot90, one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 1 --separate_G 'False'
# Rot90, separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 1 --separate_G 'True'
# Rot180, one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 2 --separate_G 'False'
# Rot180, separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 2 --separate_G 'True'
# Identity + noise (var 0.1), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 4 --noise_var 0.1 --separate_G 'False'
# Identity + noise (var 0.1), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 4 --noise_var 0.1 --separate_G 'True'
# Rot90 + noise (var 0.1), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 5 --noise_var 0.1 --separate_G 'False'
# Rot90 + noise (var 0.1), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 5 --noise_var 0.1 --separate_G 'True'
# Rot180 + noise (var 0.1), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 6 --noise_var 0.1 --separate_G 'False'
# Rot180 + noise (var 0.1), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 6 --noise_var 0.1 --separate_G 'True'
# Identity + noise (var 0.05), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 4 --noise_var 0.05 --separate_G 'False'
# Identity + noise (var 0.05), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 4 --noise_var 0.05 --separate_G 'True'
# Rot90 + noise (var 0.05), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 5 --noise_var 0.05 --separate_G 'False'
# Rot90 + noise (var 0.05), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 5 --noise_var 0.05 --separate_G 'True'
# Rot180 + noise (var 0.05), one G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 6 --noise_var 0.05 --separate_G 'False'
# Rot180 + noise (var 0.05), separate G
python usps_to_mnist_main.py --lambda_gan 1.7 --geometry 6 --noise_var 0.05 --separate_G 'True'
