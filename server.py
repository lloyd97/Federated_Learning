#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install --upgrade wandb')


# In[6]:


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
import clients


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_API_KEY"] = "183c1a6a36cbdf0405f5baacb72690845ecc8573"


# In[8]:



class Server:
    import clients
    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 optimizer_conf: Dict,
                 n_client: int = 10,
                 chosen_prob: float = 0.8,
                 local_batch_size: int = 8,
                 local_epochs: int = 10) -> None:

        # global model info
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_conf = optimizer_conf
        self.n_client = n_client
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.total_data = 0

        # create clients
        self.client_pool: List[clients.Client] = []
        self.create_client()
        self.chosen_prob = chosen_prob
        self.avg_loss = 0
        self.avg_acc = 0

    def create_client(self):
        import clients
        # this function is reusable, so reset client pool is needed
        self.client_pool: List[clients.Client] = []
        self.total_data = 0

        for i in range(self.n_client):
            model = copy.deepcopy(self.model)
            new_client = clients.Client(client_id=i,
                                model=model,
                                loss=self.loss,
                                optimizer=self.optimizer,
                                optimizer_conf=self.optimizer_conf,
                                batch_size=self.local_batch_size,
                                epochs=self.local_epochs,
                                server=self)
            self.client_pool.append(new_client)

    def broadcast(self):
        model_state_dict = copy.deepcopy(self.model.state_dict())
        for client in self.client_pool:
            client.model.load_state_dict(model_state_dict)

    def aggregate(self):
        self.avg_loss = 0
        self.avg_acc = 0
        chosen_clients = random.sample(self.client_pool,
                                       int(len(self.client_pool) * self.chosen_prob))

        global_model_weights = copy.deepcopy(self.model.state_dict())
        for key in global_model_weights:
            global_model_weights[key] = torch.zeros_like(
                global_model_weights[key])

        for client in chosen_clients:
            client.update_weights()
            print(f"Client {client.client_id}: Acc {client.accuracy}, Loss: {client.total_loss}")
            self.avg_loss += 1 / len(chosen_clients) * client.total_loss
            self.avg_acc += 1 / len(chosen_clients) * client.accuracy
            local_model_weights = copy.deepcopy(client.model.state_dict())
            for key in global_model_weights:
                global_model_weights[key] += 1 / len(chosen_clients) * local_model_weights[key]

        self.model.load_state_dict(global_model_weights)


# In[ ]:





# In[ ]:




