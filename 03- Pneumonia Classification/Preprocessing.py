#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[1]:


get_ipython().system('pip install pydicom')


# In[2]:


import os
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import cv2
import pydicom


# ### Load Dataset from Google Drive

# In[3]:


from google.colab import drive
drive.mount("/content/gdrive")


# In[4]:


get_ipython().system('unzip -q -o /content/gdrive/MyDrive/Colab\\ Notebooks/Dataset/rsna-pneumonia-detection-challenge.zip -d datasets')


# ## Train Test Data

# In[5]:


labels = pd.read_csv("/content/datasets/stage_2_train_labels.csv")


# In[6]:


labels.sample(4)


# In[7]:


len(labels)


# In[8]:


# remove duplicates
labels = labels.drop_duplicates("patientId")


# In[9]:


len(labels)


# In[10]:


ROOT_PATH = Path("/content/datasets/stage_2_train_images/")
SAVE_PATH = Path("/content/Processed")


# ## Display some Images from Train

# In[11]:


fig, axis = plt.subplots(3,3, figsize=(10,10))

counter = 0
for row in range(3):
  for column in range(3):

    patient_id = labels["patientId"].iloc[counter]
    dcm_path = ROOT_PATH/patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    dcm = pydicom.read_file(dcm_path).pixel_array

    label = labels["Target"].iloc[counter]

    axis[row][column].imshow(dcm, cmap="bone")
    axis[row][column].set_title(label)

    counter += 1


# ## Preprocessing

# In[12]:


sums, sums_squared = 0, 0

for counter,patient_id in enumerate(tqdm(labels["patientId"])):

  patient_id = labels["patientId"].iloc[counter]
  dcm_path = ROOT_PATH/patient_id
  dcm_path = dcm_path.with_suffix(".dcm")
  # Normalize
  dcm = pydicom.read_file(dcm_path).pixel_array / 255

  # Resize
  dcm_array = cv2.resize(dcm, (224,224)).astype(np.float32)

  label = labels["Target"].iloc[counter]

  train_or_val = "train" if counter < 24000 else "val"

  current_save_path = SAVE_PATH/train_or_val/str(label)
  current_save_path.mkdir(parents=True, exist_ok=True)
  np.save(current_save_path/patient_id, dcm_array)

  normalizer = 224 * 224
  if train_or_val == "train":
    sums += np.sum(dcm_array) / normalizer
    sums_squared += (dcm_array ** 2).sum()  / normalizer



# In[13]:


mean = sums / 24000
std = np.sqrt((sums_squared / 24000) - mean**2)


# In[14]:


mean, std


# In[15]:


import shutil
shutil.make_archive("Processed", 'zip', "/content/Processed")


# In[16]:


get_ipython().system('cp -r "/content/Processed.zip" /content/gdrive/MyDrive/Colab\\ Notebooks/Dataset/')


# ---
