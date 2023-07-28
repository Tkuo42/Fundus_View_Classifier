#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
from os import path
import pickle as pkl
import matplotlib.pyplot as plt
import datetime
import json
# import utils
import time
import ssl
from PIL import Image
from skimage import io 

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# import nibabel as nib
# PyTorch libraries and modules
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from sklearn.model_selection import train_test_split
#import torch.optim.lr_scheduler as lr_scheduler
#from torchmetrics.classification import BinaryAUROC 

import matplotlib.pyplot as plt
import matplotlib

csv_header = 'data/csv'
header_data = 'data/samples'
# In[3]:




class FundusDataset(Dataset):
    def __init__(self, df, mode = None):
        self.df = df
        self.mode = mode 
        
        self.augmentations = ([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            #transforms.RandomRotation(degrees=(0, 15)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((256, 256)),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.47, 0.47, 0.47], [0.3033, 0.3033, 0.3033])
                                    ])

    def __len__(self):
        return len(self.df)
  

    def __getitem__(self, index):
        #label = self.df.at[index, 'LABEL']
        
        img_loc = self.df.at[index, 'key'] 
        
        try:
            img = io.imread(os.path.join(header_data, img_loc))
            
            img = ( ((img - np.min(img)) / (np.max(img) - np.min(img)))   * 255.0)
            
            img = img.astype(np.uint8)
            
            if self.mode == 'train':
                for aug in self.augmentations:
                    img = aug(img)
            img = self.transform(img)
            #else:
               # img = self.transform(img)
            return  img  
            
        except Exception as e: 
            print(f'Error loading image {img_loc}: {e}')
            return None

# Insert csv file with image names here: 
'''

Things to note: 
1. Make sure your csv file has a column with the names of the pictures in data/samples 
2. Make sure the column is named 'key,' though you can change it in code here. 

'''

df = pd.read_csv(os.path.join(csv_header, 'csv_name.csv'))



cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(cuda, device)
sys.stdout.flush()






'''
The Model 
'''

model = models.densenet201(pretrained = False)
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.classifier.in_features, 1)
)
model.load_state_dict(torch.load('best_model_path.pt'))
model.to(device)
model.eval()

datagen_test = FundusDataset(df = df.copy(), mode = 'test') 
test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=16, num_workers=2)

y_prob= []
y_pred = []
with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = Variable(inputs, requires_grad = True).to(device)

            
        output = model(inputs)
                
        sig = nn.Sigmoid()
        probs = sig(output).cpu().detach().numpy()
        '''
        print(probs)
        
        if y_prob is None:
            y_prob = probs
        else:
            y_prob = np.concatenate((y_prob, probs), axis=0)
        '''
        pred = (torch.from_numpy(probs) > 0.6).float()
        y_pred += pred
keys = {0: 'Invalid Image', 1: 'Valid Image'}
for i in range(len(y_pred)):
    y_pred[i] = keys[int(y_pred[i])]
df['LABEL'] = y_pred

# Save csv with new labels 
df.to_csv('results/all_labeled.csv')
df[df['LABEL'] == 'Valid Image'].to_csv('results/valid_images.csv')


# In[ ]:




