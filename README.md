# gcgan-project

This repository contains source code for various experiments with [generative adversarial networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network), and in particular its application to unsupervised image-to-image translation. I also aim to demonstrate the effectiveness of the geometry-consistent variant of GAN (GcGAN) proposed in [this research paper](https://arxiv.org/abs/1809.05852).

A starting point involves translating handwritten digit images from the USPS style into the MNIST style, while retaining the content of these images (i.e. such that the digit itself is invariant via the translation). Several GAN variants are mixed and matched to examine their individual and collective power in such a translation task.

# Acknowledgments

My code is sprinkled here and there with ideas from the source code repository associated with the aforementioned paper on GcGANs. It can be accessed [here](https://github.com/hufu6371/GcGAN).

I would also like to thank [Dr. Mingming Gong](https://mingming-gong.github.io/index.html) (a co-first author of the GcGAN paper) for his excellent supervision throughout the duration of this project.
