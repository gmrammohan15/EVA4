from __future__ import print_function
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
from network import *
from prepare_dataset import *
import torch.optim as optim


class MnistModel:
    def __init__(self, model,dataloader_args, lr = 0.01, momentum = 0.9, step_size = 0, gamma = 0.01):
        self.m_train_losses = []
        self.m_test_losses = []
        self.m_train_acc = []
        self.m_test_acc = []
        self.m_model=copy.deepcopy(model)
        self.m_optimizer=optim.SGD(self.m_model.parameters(), lr, momentum, weight_decay=1e-5)
        self.load_mnist_data(dataloader_args, train_data, test_data)

    def clone_model(self):
        copy.deepcopy(self.m_model)

    def load_mnist_data(self,dataloader_args,train_set,test_set):
        self.m_train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        self.m_test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)
