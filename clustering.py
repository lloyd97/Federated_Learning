#!/usr/bin/env python
# coding: utf-8

# In[14]:


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


# In[30]:


class Cluster:
    def __init__(self,
                 nr_nodes: int,
                 step: 0,
                 MAX_X: int,
                 MAX_Y: int,
                 MIN_V: 0.1,
                 MAX_V: 1,
                 MAX_WT: int,
                 STEPS_TO_IGNORE: 10000,
                 CALCULATE_CONTACTS: False,
                 RANGE: 1,
                 rw=None) -> None:
        
        self.nr_nodes = nr_nodes
        self.step = 0
        np.random.seed(0xffff)
        self.MAX_X, self.MAX_Y = 100, 100
        self.MIN_V, self.MAX_V = 0.1, 1.
        self.MAX_WT = 100.
        self.STEPS_TO_IGNORE = 10000
        self.CALCULATE_CONTACTS = False
        self.RANGE = 1.
        self.rw = mobility.random_walk(self.nr_nodes, dimensions=(self.MAX_X, self.MAX_Y))

    def coord(self,rw):
        self.step += 1
        positions = next(self.rw)
        #print(str('position: ') + str(positions))
        return positions
        


# In[ ]:




