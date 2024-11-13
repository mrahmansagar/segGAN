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
from scipy import ndimage as nd


from GANs import utils
from keras.models import load_model


val_dir = "E:\\Data\\segGAN\\ground_truth\\val\\"

patch_dir = os.path.join(val_dir, "images")
mask_dir = os.path.join(val_dir, "masks")

all_masks = utils.load_images_in_shape(mask_dir, color_mode = 'grayscale')
all_patches = utils.load_images_in_shape(patch_dir, color_mode = 'grayscale')

all_masks = (all_masks > 0).astype('float32')


plt.imshow(all_masks[0], cmap='gray')
plt.show()

model_dir = 'E:\\projects\\segGAN\\mask2patch_202411061638\\'

all_trained_model = glob(model_dir + '*.h5')

model_path = all_trained_model[99]

model = load_model(model_path)



sample_size = 5
idx = np.random.randint(0, len(all_masks), sample_size)

patches = all_patches[idx]

masks = all_masks[idx]
masks = (masks - 127.5) / 127.5

gen_patch = model.predict(masks)
gen_patch = (gen_patch + 1) / 2.0 

# Apply Median filter 
# gen_patch = nd.median_filter(gen_patch, size=2)
# Apply Gaussian blur
# sigma = 0.5  # Standard deviation for Gaussian kernel
# gen_patch = nd.gaussian_filter(gen_patch, sigma=sigma)


# #gen_patch = img_as_ubyte(gen_patch)


# ploting the images 
plt.figure(figsize=(sample_size*2, sample_size*1.5))
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(sample_size):
    plt.subplot(3, sample_size, i+1)
    plt.axis('off')
    plt.imshow(masks[i], cmap='gray')
    
    plt.subplot(3, sample_size, sample_size+1+i)
    plt.axis('off')
    #apply median filter 
    # plt.imshow(nd.median_filter(gen_patch[i], size=2), cmap='gray')
    plt.imshow(gen_patch[i], cmap='gray')

    # Display generated patches in the third row
    plt.subplot(3, sample_size, 2 * sample_size + i + 1)
    plt.axis('off')
    plt.imshow(patches[i], cmap='gray')
    
    
plt.show()



# from scipy import ndimage as nd
# gen_patch_median = nd.median_filter(gen_patch, size=3)


# plt.imshow(gen_patch_median, cmap='gray')
# plt.show()


# # Step 4: Apply Gaussian blur
# sigma = 0.5  # Standard deviation for Gaussian kernel
# gen_patch_gaussian = nd.gaussian_filter(gen_patch, sigma=sigma)

# plt.imshow(gen_patch_gaussian, cmap='gray')
# plt.show()

