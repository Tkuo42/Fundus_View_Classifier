#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
#!{sys.executable} -m pip 
#!{sys.executable} -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1



# In[2]:


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


# In[3]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def __str__(self):
        return f' is {self.avg}'
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[4]:


#hyperparameters 

do_train=True
n_iters = 75
#age_lower = 20
#age_upper = 100
disp_labels = ['Good', 'Bad'] # My classifiers are just yes/no 
print_every = 1
plot_every = 1
batch_size = 16
model_name = 'densenet121-finetuned' # Can test out different pretrained models 
save_dir = 'Models/' 
metric_name = 'accuracy' # For me it would be different -- i put accuracy but we will see if that is an option
predictor = 'middle_sagittal' # Whether the image is good or not 
lr = 0.0001
maximize_metric=True
patience = 5
early_stop=False
prev_val_loss = 1e10
itr = 0
best_model_dir = None#'/media/Datacenter_storage/BiologicalAge/Code/Models/train/train-23/'
header_data = '/media/Datacenter_storage/Kowa_Images_rgb/sample_1000/' #Kowa_Images_rgb/sample_1000.csv
csv_header = '/home/tyler'
# Remember to use absolute paths! 


# In[5]:


# Change this to my own type of dataset 

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
        label = self.df.at[index, 'LABEL']
        
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
            return  img , label 
            
        except Exception as e: 
            print(f'Error loading image {img_loc}: {e}')
            return None


# In[6]:


df = pd.read_csv(os.path.join(csv_header + '/good_sample_1000.csv'))


# In[7]:


with open('fundus_data_new.pkl', 'rb') as fp:
    fundus_data = pkl.load(fp)
df_train = fundus_data['train'].reset_index()
df_test = fundus_data['test'].reset_index()
df_val = fundus_data['val'].reset_index()

df_test['LABEL'][df_test['LABEL'] == True].count()


# In[8]:


classes = (True, False)


# In[9]:


model = models.densenet201(pretrained = False)
print(model)
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.classifier.in_features, 1)
)


# In[10]:


datagen_train = FundusDataset(df =  df_train.copy(), mode = 'train') 
datagen_val = FundusDataset(df = df_val.copy(), mode = 'val') 
datagen_test = FundusDataset(df = df_test.copy(), mode = 'test') 


# In[11]:


class_weights = {True: 2.75, False: 1.0}
train_weights = [class_weights[e] for e in df_train["LABEL"]]


sampler = WeightedRandomSampler(train_weights, num_samples = len(df_train))


# In[12]:


train_loader = DataLoader(dataset=datagen_train,  batch_size=batch_size, num_workers=2, sampler = sampler )
val_loader = DataLoader(dataset=datagen_val,  shuffle=False, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=batch_size, num_workers=2)


# In[13]:


#weights = torch.tensor(5)


# In[14]:


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(cuda, device)
sys.stdout.flush()
criterion = torch.nn.BCEWithLogitsLoss().to(device)   #BCEWithLogitsLoss().to(device)
#pos_weight = torch.from_numpy(np.array([3, 1])
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)


# In[15]:


current_loss = 0
min_valid_loss = np.inf
all_losses = []
best_model = None 


# In[16]:


torch.cuda.empty_cache()
if cuda:
    model = model.to(device)
    model.train()


# # Did you change the model path? And mark what you did differently this model? 

# In[17]:


df_loss = pd.DataFrame(columns = ['train', 'val', 'train_auroc', 'val_auroc'])
prev_val_acc = 0
patience_val = 0 
min_val_acc = np.inf 
print('start')
sys.stdout.flush()
for epoch in range(n_iters):
    
    y_pred = []
    y_true = []
    y_prob= None
    
    losses = AverageMeter()
    valid_losses = AverageMeter()
    
    
    model.train()
    
    for inputs, labels in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        model.train()
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs, labels = Variable(inputs, requires_grad = True).to(device), Variable(labels, requires_grad = False).to(device)
        
        labels = labels.unsqueeze(1)
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = model(inputs)
        sig = nn.Sigmoid()
        probs = sig(output).cpu().detach().numpy()  # (batch_size, )
        if y_prob is None:
            y_prob = probs
        else:
            y_prob = np.concatenate((y_prob, probs), axis=0)
        y_true += list(labels.cpu().detach().numpy().astype(int))
        
        loss = criterion(output, labels)
      
        losses.update(loss.item(), inputs.size(0))
      
        loss.backward()
        optimizer.step()
    
    
     
    print(f'Iteration {epoch}...')
    print(f'Training Loss{str(losses)}')            
    df_loss.at[epoch, 'train'] = losses.avg    
    
    roc_score = roc_auc_score(y_true, y_prob)
    
    model.eval() 
    print(f'Training Auroc: {roc_score}')
    df_loss.at[epoch, 'train_auroc']= roc_score
    
    y_pred = []
    y_true = []
    y_prob= None
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader): 
            labels = labels.type(torch.LongTensor)
            inputs, labels = Variable(inputs, requires_grad = True).to(device), Variable(labels, requires_grad = False).to(device)
            labels = labels.unsqueeze(1)
            labels = labels.float()
            
            output = model(inputs)
                
            sig = nn.Sigmoid()
            probs = sig(output).cpu().detach().numpy()
           # y_pred += list(preds)
            
            
                
            if y_prob is None:
                y_prob = probs
            else:
                y_prob = np.concatenate((y_prob, probs), axis=0)
            y_true += list(labels.cpu().detach().numpy().astype(int))
            loss = criterion(output, labels)
            valid_losses.update(loss.item(), inputs.size(0))
            
    
    
    
    print(f'Validation loss{str(valid_losses)}')
    
    df_loss.at[epoch, 'val'] = valid_losses.avg
    roc_score = roc_auc_score(y_true, y_prob)
    print(f'Validation Auroc: {roc_score}')
    df_loss.at[epoch, 'val_auroc']= roc_score
    
    if valid_losses.avg < prev_val_acc: 
        patience_val = 0
    if valid_losses.avg >= prev_val_acc: 
        patience_val += 1 
    if patience_val > patience: 
        break 
    prev_val_acc = valid_losses.avg
    
    if  valid_losses.avg < min_val_acc:
        min_val_acc = valid_losses.avg
        torch.save(model.state_dict(), 'Models/model2/model_2_best.pt')

   
print('Finished Training')
display(df_loss)


# In[18]:


df_loss


# In[19]:


plt.plot(range(0, len(df_loss) ), df_loss['val_auroc'], c = 'red')
plt.plot(range(0, len(df_loss)), df_loss['train_auroc'], c = 'blue')

plt.legend(['Val_acc', 'Train_acc'], loc = 3)
plt.title('AUROC Over Epoch')
plt.ylabel('AUROC')
plt.xlabel('Epoch')
plt.savefig('Models/model2/model2_AUROC.png')
plt.show()


# In[20]:


plt.plot(range(len(df_loss)), df_loss['train'], c = 'green')
plt.plot(range(len(df_loss)), df_loss['val'], c = 'orange')
plt.ylim(bottom = 0)
plt.legend([ 'train_loss', 'val_loss'], loc = 3)
plt.title('Loss over Epoch')
plt.ylabel('BCELoss')
plt.xlabel('Epoch')
plt.savefig('Models/model2/model2.BCELoss.png')
plt.show()


# # Remember to do test dataset as well 

# In[21]:


model = models.densenet201(pretrained = False)
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.classifier.in_features, 1)
)
model.load_state_dict(torch.load('Models/model2/model_2_best.pt'))
model.to(device)
model.eval()


# In[22]:


y_pred = []
y_true = []
y_prob= None
test_losses = AverageMeter()


with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        labels = labels.type(torch.LongTensor)
        inputs, labels = Variable(inputs, requires_grad = True).to(device), Variable(labels, requires_grad = False).to(device)
        labels = labels.unsqueeze(1)
        labels = labels.float()
            
        output = model(inputs)
                
        sig = nn.Sigmoid()
        probs = sig(output).cpu().detach().numpy()
        
        if y_prob is None:
            y_prob = probs
        else:
            y_prob = np.concatenate((y_prob, probs), axis=0)
        
        pred = (torch.from_numpy(probs) > 0.5).float()
        y_pred += pred
        y_true += list(labels.cpu().detach().numpy().astype(int))
        loss = criterion(output, labels)
        test_losses.update(loss.item(), inputs.size(0))

        
roc_score = roc_auc_score(y_true, y_prob)

print(f'ROC Score: {roc_score}')
print(f'Loss Score: {test_losses.avg}')

plt.bar(['AUROC', 'BCELoss'],[roc_score, test_losses.avg])
plt.savefig(os.path.join('Models/model2', 'stats.png'))
plt.show()


cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=np.array(['0', '1']))
disp.plot()
plt.savefig(os.path.join('Models/model2', "confusion_matrix.png"))


# In[23]:


df_true = pd.DataFrame(y_true)
df_true
df_test['LABEL'][54]


# In[24]:


df_prob = pd.DataFrame(y_prob)


# In[25]:


b = df_prob[df_prob[0] > 0.6]


# In[26]:


a= df_true[df_true[0] == 0]


# In[27]:


a = (a).dropna()
b = b.dropna()
(a + b).dropna()


# In[29]:


img = io.imread(os.path.join(header_data, df_test['key'][139]))
print(df_test["key"][139])

plt.imshow(img)
#print(df_test['LABEL'][178])


# In[ ]:


# Me: 3, 6, 14, 22, 26, 29, 30, 44, 47, 54, 72, 73, 86, 96, 99, 102, 103, 113, 120, 122, 123, 145, 156   
# Ami: 26?
# Bhavik: 26?
# Model Wrong: 62, 147, 178 


# In[ ]:


# Me: 
# Model Wrong: 31, 40, 110, 116, 148, 192 


# In[ ]:


c = df_prob[df_prob[0] < 0.4]
d = df_true[df_true[0] == 1]

c = c.dropna()
d = d.dropna()

(c + d).dropna()


# In[ ]:


df_test['LABEL'][df_test['LABEL'] == True].count()


# In[ ]:


img = io.imread(os.path.join(header_data, df_test['key'][192]))

plt.imshow(img)
print(df_test["key"][192])
print(df_test['LABEL'][192])


# In[ ]:


df_val['LABEL'][df_val['LABEL'] == True].count()


# In[ ]:




