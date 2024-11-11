# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:03:25 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Training a pix2pix for segmentation only using the synthetic data

"""

import os 
os.sys.path.insert(0, 'E:\\dev\packages')

#from glob import glob
#from tqdm import tqdm

#import numpy as np
import matplotlib.pyplot as plt

from GANs import utils
from GANs.pix2pix import models

#from PIL import Image


root_dir = "E:\\Data\\segGAN\\"

data_dir = os.path.join(root_dir, "generated\\")

mskNpatch = utils.load_images_in_shape(data_dir, color_mode = 'grayscale')

plt.imshow(mskNpatch[0, :, :, 0], cmap='gray')
plt.show()

src_patch = mskNpatch[:, :, 256+10:512+10, :]
tar_msk = mskNpatch[:, :, 0:256, :]

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_patch[i].astype('uint8'), cmap='gray')
    
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_msk[i].astype('uint8'), cmap='gray')
plt.show()

# scalling the data between -1 to 1
src_patch = (src_patch - 127.5) / 127.5
tar_msk = (tar_msk - 127.5) / 127.5


src_shape = src_patch.shape[1:]
tar_shape = tar_msk.shape[1:]

tar_channel = tar_msk.shape[-1]


dis = models.build_discriminator(src_shape=src_shape, tar_shape=tar_shape)
gen = models.build_generator(input_shape=src_shape, output_channel=tar_channel)


p2p_model = models.build_pix2pix(gen, dis)


# train 
models.train_pix2pix(gen, dis, p2p_model, src_patch, tar_msk, epochs=1000, summary_interval=10, name='patch2msk')

