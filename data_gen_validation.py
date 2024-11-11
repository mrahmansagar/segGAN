# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:34:25 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Model outcome with real mask  
 
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from GANs import utils
from keras.models import load_model


val_dir = "E:\\Data\\segGAN\\val\\"

patch_dir = os.path.join(val_dir, "images")
mask_dir = os.path.join(val_dir, "masks")

masks = utils.load_images_in_shape(mask_dir, color_mode = 'grayscale')
patches = utils.load_images_in_shape(patch_dir, color_mode = 'grayscale')

masks = (masks > 0).astype('float32')


plt.imshow(masks[0], cmap='gray')
plt.show()


masks = (masks - 127.5) / 127.5


model_dir = 'E:\\projects\\segGAN\\mask2patch_202411061638\\'

all_trained_model = glob(model_dir + '*.h5')


model_path = all_trained_model[99]

model = load_model(model_path)

gen_patch = model.predict(masks)

gen_patch = (gen_patch + 1) / 2.0 

gen_patch = img_as_ubyte(np.squeeze(gen_patch[0]))


plt.imshow(gen_patch, cmap='gray')
plt.show()


plt.imshow(patches[0], cmap='gray')
plt.show()


from scipy import ndimage as nd
gen_patch_median = nd.median_filter(gen_patch, size=3)


plt.imshow(gen_patch_median, cmap='gray')
plt.show()


# Step 4: Apply Gaussian blur
sigma = 0.5  # Standard deviation for Gaussian kernel
gen_patch_gaussian = nd.gaussian_filter(gen_patch, sigma=sigma)

plt.imshow(gen_patch_gaussian, cmap='gray')
plt.show()

