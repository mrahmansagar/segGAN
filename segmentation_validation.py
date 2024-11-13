# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:54:43 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

# Segmentation  
Testing the performance of trained model with sythetic data on the real data 

 
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


model_dir = 'E:\\projects\\segGAN\\patch2msk_202411111041\\'

all_trained_model = glob(model_dir + '*.h5')

model_path = all_trained_model[0]

model = load_model(model_path)


sample_size = 7
# idx = np.random.randint(0, len(all_patches), sample_size)
idx = np.arange(sample_size)

patches = all_patches[idx]
patches = (patches - 127.5) / 127.5

masks = all_masks[idx]

gen_masks = model.predict(patches)
#gen_masks = (gen_masks + 1) / 2.0
# gen_masks = gen_masks > 0 
# gen_masks = nd.binary_erosion(gen_masks)
# gen_masks = img_as_ubyte(gen_masks)
# gen_masks = gen_masks.astype('float32')

# ploting the images 
plt.figure(figsize=(sample_size*2, sample_size*1.5))
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(sample_size):
    plt.subplot(3, sample_size, i+1)
    plt.axis('off')
    plt.imshow(patches[i], cmap='gray')
    
    plt.subplot(3, sample_size, sample_size+1+i)
    plt.axis('off')
    #apply median filter 
    # plt.imshow(nd.median_filter(gen_patch[i], size=2), cmap='gray')
    plt.imshow(gen_masks[i], cmap='gray')

    # Display generated patches in the third row
    plt.subplot(3, sample_size, 2 * sample_size + i + 1)
    plt.axis('off')
    plt.imshow(masks[i], cmap='gray')
    
    
plt.show()