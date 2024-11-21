# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:53:40 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Generating 3D masks with porespy

"""
import os 
os.sys.path.insert(0, "E:\\dev\\packages")
import proUtils.utils as pu

import numpy as np
import matplotlib.pyplot as plt

import porespy as ps
from skimage import img_as_ubyte
from scipy import ndimage as nd

from keras.models import load_model
from keras.utils import img_to_array

from glob import glob


# Blob generation 3D
shape = [256, 256, 256]
porosity = 0.3
blobiness = 0.7

ps_vol = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)

ps_vol = img_as_ubyte(ps_vol)

pu.save_vol_as_slices(ps_vol, 'output/ps_msk3D')


model_dir = 'E:\\projects\\segGAN\\mask2patch_202411061638\\'

all_trained_model = glob(model_dir + '*.h5')

selected_model = 0

model_path = all_trained_model[selected_model]

model = load_model(model_path)


ps_patch3d = np.empty_like(ps_vol)

for i in range(0,256):
    fake_msk = img_to_array(ps_vol[i, :, :])
    fake_msk_scaled = (fake_msk - 127.5) / 127.5
    fake_msk_in_shape = np.expand_dims(fake_msk_scaled, axis=0)
    
    gen_patch = model.predict(fake_msk_in_shape)
    gen_patch = (gen_patch + 1) / 2.0 
    gen_patch = img_as_ubyte(np.squeeze(gen_patch))
    gen_patch_median = nd.median_filter(gen_patch, size=2)
    
    ps_patch3d[i, :, :] = gen_patch_median
    

pu.save_vol_as_slices(ps_patch3d, 'output/ps_patch3D')

