# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:52:56 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Data preprocessing and training of pix2pix network for data generation 
"""


import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

from glob import glob
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt

from GANs import utils
from GANs.pix2pix import models


from PIL import Image


root_dir = "E:\\Data\\segGAN\\"

image_dir = os.path.join(root_dir, "ground_truth\\images\\")
mask_dir = os.path.join(root_dir, "ground_truth\\masks\\")

src_img = utils.load_images_in_shape(mask_dir, color_mode = 'grayscale')
tar_img = utils.load_images_in_shape(image_dir, color_mode = 'grayscale')

src_img = (src_img > 0).astype('float32')


print('loaded', src_img.shape, 'source images')
print('loaded', tar_img.shape, 'target images')

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_img[i].astype('uint8'), cmap='gray')
    
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_img[i].astype('uint8'), cmap='gray')
plt.show()


# scalling the data between -1 to 1
src_mask = (src_img - 127.5) / 127.5
tar_patch = (tar_img - 127.5) / 127.5


src_shape = src_mask.shape[1:]
tar_shape = tar_patch.shape[1:]

tar_channel = tar_patch.shape[-1]


dis = models.build_discriminator(src_shape=src_shape, tar_shape=tar_shape)
gen = models.build_generator(input_shape=src_shape, output_channel=tar_channel)


p2p_model = models.build_pix2pix(gen, dis)


# train 
models.train_pix2pix(gen, dis, p2p_model, src_mask, tar_patch, epochs=1000, summary_interval=10, name='mask2patch')
