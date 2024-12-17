# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:57:43 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
 
Fine Tune Segment Anything Model 
"""

import os
os.sys.path.insert(0, "E:\\dev\\packages")

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import img_as_ubyte
# from scipy import ndimage as nd

import random


root_dir = "E:\\Data\\segGAN\\"

image_dir = glob("E:\\Data\\segGAN\\ground_truth\\train\\images\\"+ "*")
mask_dir = glob("E:\\Data\\segGAN\\ground_truth\\train\\masks\\"+ "*")

patches = []
for im_dir in image_dir:
    im = Image.open(im_dir)
    im = np.asarray(im)
    patches.append(im)

patches = np.asarray(patches)

masks = []
for msk_dir in mask_dir:
    msk = Image.open(msk_dir)
    msk = np.asarray(msk)
    msk = img_as_ubyte(msk > 0)
    masks.append(msk)

masks = np.asarray(masks)
# masks = (masks > 0).astype('uint8')

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(patches[i].astype('uint8'), cmap='gray')
    
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(masks[i].astype('uint8'), cmap='gray')
plt.show()


print('loaded', patches.shape, 'patches')
print('loaded', masks.shape, 'masks')


from datasets import Dataset

dataset_dict = {
    "image": [Image.fromarray(img) for img in patches],
    "label": [Image.fromarray(mask) for mask in masks],
    }

dataset = Dataset.from_dict(dataset_dict)


#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


img_num = random.randint(0, patches.shape[0]-1)
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]
print(np.array(example_mask).shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

bounding_box = get_bounding_box(np.array(example_mask))
print(bounding_box)
print(np.array(example_mask).shape)
plt.imshow(example_mask, cmap='gray')
plt.plot([bounding_box[0], bounding_box[2]], [bounding_box[1], bounding_box[1]], 'r')  # Top line
plt.plot([bounding_box[0], bounding_box[2]], [bounding_box[3], bounding_box[3]], 'r')  # Bottom line
plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[1], bounding_box[3]], 'r')  # Left line
plt.plot([bounding_box[2], bounding_box[2]], [bounding_box[1], bounding_box[3]], 'r')  # Right line
plt.axis('off')
plt.show()



from torch.utils.data import Dataset

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    image = np.array(image)

    if image.ndim == 2:
      image = np.expand_dims(image, axis=-1)
      image = np.repeat(image, 3, axis=2)
      
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)
  
# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)
  
batch["ground_truth_mask"].shape


# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
    

from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

#Training loop
num_epochs = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    
    
# Save the model's state dictionary to a file
torch.save(model.state_dict(), "SAM/model_checkpoint.pth")


from transformers import SamModel, SamConfig, SamProcessor
import torch

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_mito_model.load_state_dict(torch.load("SAM/model_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)


import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# let's take a random training example
idx = random.randint(0, patches.shape[0]-1)

# load image
test_image = dataset[idx]["image"]

test_image = np.array(test_image)

if test_image.ndim == 2:
  test_image = np.expand_dims(test_image, axis=-1)
  test_image = np.repeat(test_image, 3, axis=2)

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

my_mito_model.eval()

# forward pass
with torch.no_grad():
    outputs = my_mito_model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the first image on the left
axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(medsam_seg, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Plot the second image on the right
axes[2].imshow(medsam_seg_prob)  # Assuming the second image is grayscale
axes[2].set_title("Probability Map")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()
