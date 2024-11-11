# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:29:47 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Using the trained model to create patches with porespy masks

"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import proUtils.utils as pu


from glob import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from scipy import ndimage as nd

from GANs import utils
from keras.models import load_model
from keras.utils import img_to_array

import porespy as ps

import tifffile


save_dir = 'E:\\Data\\segGAN\\generated\\'

model_dir = 'E:\\projects\\segGAN\\mask2patch_202411061638\\'

all_trained_model = glob(model_dir + '*.h5')

selected_model = 99

model_path = all_trained_model[selected_model]

model = load_model(model_path)

# Parameters for synthetic image generation
shape = [256, 256]  # 3D image dimensions (x, y, z)
porosity_range = [0.3, 0.7]  # Desired porosity value
blobiness_range = [0.4, 2.4]  # Blobiness parameter for blob generation

# Creating fake mask and corresponding generated patch 

for seed in range(0, 5500):
    # random porosity between range 
    porosity = random.uniform(porosity_range[0], porosity_range[1])
    porosity = float(f'{porosity:.2f}')
    
    blobiness = random.uniform(blobiness_range[0], blobiness_range[1])
    blobiness = float(f'{blobiness:.2f}')
    ps_img = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness, seed=seed)
    
    fake_msk = img_to_array(ps_img)
    fake_msk_scaled = (fake_msk - 127.5) / 127.5
    fake_msk_in_shape = np.expand_dims(fake_msk_scaled, axis=0)
    
    gen_patch = model.predict(fake_msk_in_shape)
    gen_patch = (gen_patch + 1) / 2.0 
    gen_patch = img_as_ubyte(np.squeeze(gen_patch[0]))
    gen_patch_median = nd.median_filter(gen_patch, size=2)
    
    ps_img_uint8 = img_as_ubyte(ps_img)
    
    maskNpatch = pu.combine_ndarrays(ps_img_uint8, gen_patch_median)
    
    fname = os.path.join(save_dir, f'mskNpatch_s{seed}_p{porosity}_b{blobiness}.tif')
    
    tifffile.imwrite(fname, maskNpatch)
    





# # Step 1: Generate 3D synthetic image using overlapping spheres or blobs
# binary_image = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)


# # plt.imshow(binary_image, cmap='gray')
# # plt.show()



# model_dir = 'E:\\projects\\segGAN\\mask2patch_202411061638\\'

# all_trained_model = glob(model_dir + '*.h5')


# fake_msk = img_to_array(binary_image)
# fake_msk_scaled = (fake_msk - 127.5) / 127.5
# fake_msk_in_shape = np.expand_dims(fake_msk_scaled, axis=0)


# model_path = all_trained_model[99]

# model = load_model(model_path)

# gen_patch = model.predict(fake_msk_in_shape)

# gen_patch = (gen_patch + 1) / 2.0 

# gen_patch = img_as_ubyte(np.squeeze(gen_patch[0]))

# plt.imshow(binary_image, cmap='gray')
# plt.show()


# plt.imshow(gen_patch, cmap='gray')
# plt.show()


# from scipy import ndimage as nd
# gen_patch_median = nd.median_filter(gen_patch, size=3)

# plt.imshow(gen_patch_median, cmap='gray')
# plt.show()