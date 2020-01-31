#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 00:05:00
#SBATCH --cpus-per-task=1

python usps_to_mnist_main.py
