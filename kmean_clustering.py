#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import numpy as np
import logging
import mobility
from scipy.spatial.distance import cdist
from mobility import gauss_markov, reference_point_group,     tvc, truncated_levy_walk, random_direction, random_waypoint, random_walk

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
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.cluster import KMeans


# In[6]:


class Kmeans_cluster:
    def __init__(self,
                 coordinate: list,
                 best_k: int,
                 n_leader: int,
                 cluster_coords: list,
                 kmeans: None,
                 client_list: list,
                 n_client: int) -> None:
        
        self.coordinate = coordinate
        self.best_k = best_k
        self.n_leader = n_leader
        self.cluster_coords = cluster_coords
        self.kmeans =  None
        self.client_list = []
        self.n_client = n_client
       
    #Creating cluster
    def getElbowCurve(self,coordinate):
        self.best_k = 3
        self.n_leader = self.best_k
        distortions = []
        K = range(1,10)

        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(coordinate)
            distortions.append(kmeanModel.inertia_)

        #print(distortions)
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        return self.best_k

    def createCluster(self,n_leader,coordinate):
        self.kmeans = KMeans(n_clusters=n_leader)
        self.kmeans.fit(coordinate)

        #print(str('Center coordinate of each cluster')+str(kmeans.cluster_centers_))
        print(str('labels of clusters: ')+str(self.kmeans.labels_))
        self.plotCluster(self.kmeans,coordinate)
        return self.kmeans

    def plotCluster(self,kmeans,coordinate):
        plt.scatter(coordinate[:,0],coordinate[:,1], c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

    def initiaiseClusterCoordinates(self,kmeans):
        cluster_coords = [[] for _ in range(self.n_leader)]
        for i in range(int(self.n_leader)):    
            for j in range(len(self.kmeans.labels_)):
                if (self.kmeans.labels_[j] == i):
                    cluster_coords[i].append(j)
        return cluster_coords

    def getClientList(self,cluster_coords):
        client_size = cluster_coords
        
        for i in range(int(self.n_leader)):
            self.client_list.append(len(client_size[i]))
        return self.client_list

    def fillingNoneValue(self,cluster_coords):
        for i in range(int(self.best_k)):
            for j in range(int(self.n_client - len(cluster_coords[i]))):
                cluster_coords[i].append(None)
        return cluster_coords


# In[ ]:




