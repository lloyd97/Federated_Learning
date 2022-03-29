#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade wandb')


# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import os
from typing import Any, Dict, List
import copy
import random
import wandb


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_API_KEY"] = "183c1a6a36cbdf0405f5baacb72690845ecc8573"


# In[22]:


class Client:
    def __init__(self,
                 client_id: Any,
                 model: torch.nn.Module,
                 loss: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 optimizer_conf: Dict,
                 batch_size: int,
                 epochs: int,
                 server=None) -> None:
        self.client_id = client_id
        self.model = model
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters(), **optimizer_conf)
        self.batch_size = batch_size
        self.epochs = epochs
        self.server = server
        self.accuracy = None
        self.total_loss = None

        self.data = None
        self.data_loader = None
        
        self.actual_Xcoord = 0
        self.actual_Ycoord = 0
        self.previous_Xcoord = 0
        self.previous_Ycoord = 0
        
    def setXCoordinate(self, x):
        self.actual_Xcoord = x
    
    def setYCoordinate(self, y):
        self.actual_Ycoord = y
        
    def setPreviousXCoordinate(self, x):
        self.previous_Xcoord = x
        
    def setPreviousXCoordinate(self, y):
        self.previous_Ycoord = y
        
    def setData(self, data):
        self.data = data
        self.data_loader = torch.utils.data.DataLoader(self.data,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        self.server.total_data += len(self.data)

    def update_weights(self):
        for eps in range(self.epochs):
            total_loss = 0
            total_batches = 0
            total_correct = 0

            for _, (feature, label) in enumerate(self.data_loader):
                feature = feature.to(device)
                label = label.to(device)
                #print(str('feature ')+str(feature))
                #print(str('label ')+str(label))
                
                y_pred = self.model(feature)
                y_pred_decode = torch.argmax(y_pred, dim=1)
                #print(str('y_pred ')+str(y_pred))
                #print(str('y_pred_decode ')+str(y_pred_decode))
                
                total_correct += y_pred_decode.eq(label).sum().item()
                loss = self.loss(y_pred, label)
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
                

            self.total_loss = total_loss / total_batches
            self.accuracy = total_correct / (total_batches * self.batch_size)


# In[ ]:




