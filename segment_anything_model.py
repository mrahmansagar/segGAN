# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:06:12 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Segment Anything Model 
"""

import os
os.sys.path.insert(0, "E:\\dev\\packages")

from glob import glob
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


from PIL import Image 

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


val_dir = "E:\\Data\\segGAN\\ground_truth\\val\\"

patch_dir = glob(os.path.join(val_dir, "images") + "\\*")
mask_dir = glob(os.path.join(val_dir, "masks") + "\\*")


aPatch = Image.open(patch_dir[1])
aPatch = np.asarray(aPatch)

plt.imshow(aPatch, cmap='gray')
plt.axis('off')
plt.show()

sam_checkpoint = "SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=128,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

image = np.expand_dims(aPatch, axis=2)
image = np.repeat(image, 3, axis=2)

masks = mask_generator_.generate(image)

print(len(masks))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns[1:]:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        

plt.figure()
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
