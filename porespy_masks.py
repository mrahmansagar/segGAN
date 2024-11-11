# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:45:54 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Generating masks with porespy

"""

import os 
os.sys.path.insert(0, "E:\\dev\\packages")
import proUtils.utils as pu


import numpy as np
import matplotlib.pyplot as plt


import porespy as ps




# Blob generation 
shape = [256, 256]


blobiness_array = np.arange(20, 30) /10  # Array of blobiness values
porosity_array = np.arange(1, 10) / 10   # Array of porosity values

# Create figure and axes for plotting based on the lengths of blobiness and porosity arrays
fig, axes = plt.subplots(len(porosity_array), len(blobiness_array), figsize=(40, 40))

# Generate and plot each image with different blobiness and porosity
for row, porosity in enumerate(porosity_array):
    for col, blobiness in enumerate(blobiness_array):
        ps_img = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
        axes[row, col].imshow(ps_img, cmap='gray')
        axes[row, col].set_title(f'Blobiness: {blobiness:.1f}', fontsize=40)
        
        # Hide ticks and spines for cleaner look
        axes[row, col].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].spines['right'].set_visible(False)
        axes[row, col].spines['bottom'].set_visible(False)
        axes[row, col].spines['left'].set_visible(False)
        
        # Add y-axis label on the leftmost column only
        if col == 0:
            axes[row, col].set_ylabel(f'Porosity: {porosity:.1f}', fontsize=40)

# Adjust layout
plt.tight_layout()
plt.savefig('blobiness2-2p9.png')
plt.show()


import tifffile
ps_img = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness, seed=10000)
tifffile.imwrite('ps_img.tif', ps_img)
