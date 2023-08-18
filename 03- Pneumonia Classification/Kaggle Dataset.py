#!/usr/bin/env python
# coding: utf-8

# # Kaggle Dataset

# [https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/)

# In[2]:


import os
import sys


# In[ ]:


get_ipython().system('pip install -q kaggle')


# In[ ]:


os.environ["KAGGLE_USERNAME"] = "zahedgolabi"
os.environ["KAGGLE_KEY"] = "#########################"


# In[ ]:


get_ipython().system('kaggle competitions download -c rsna-pneumonia-detection-challenge')


# ## Save Dataset to Google Drive

# In[ ]:


from google.colab import drive
drive.mount("/content/gdrive")


# In[ ]:


get_ipython().system('cp -r "/content/rsna-pneumonia-detection-challenge.zip" /content/gdrive/MyDrive/Colab\\ Notebooks/Dataset/')


# In[ ]:


get_ipython().system('unzip -q -o rsna-pneumonia-detection-challenge.zip -d datasets')

