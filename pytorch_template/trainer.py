import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import yaml
import numpy as np
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from pytorch_template.utils import *

from sklearn.neighbors import KNeighborsClassifier

CPU = torch.device("cpu")  

class Trainer:
    def __init__(self, device:torch.device, model: nn.Module, criterion, optimizer: nn.Module, cfg: MyConfig):
        self.__model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.calc_acc = Accuracy()
        self.cfg = cfg
        self.device = device

    @property
    def model(self):
        return self.__model

    def calc_scores(self, output, y):
        loss = self.criterion(output, y)
        with torch.no_grad():
            pred_y = output.argmax(dim=1, keepdim=True).to(CPU).detach().numpy()
            y = y.to(CPU).detach().numpy()
            acc = self.calc_acc(pred_y, y)
        return loss, acc

    def train_step(self, train_loader: DataLoader) -> tuple[float, float]:
        avg_loss = AvgMeter()
        avg_acc = AvgMeter()
        for train_X, train_y in tqdm(train_loader, desc="Epoch", leave=False, total=len(train_loader)):        
            train_X = train_X.to(self.device)
            train_y = train_y.to(self.device)

            output = self.__model(train_X)
            loss, acc = self.calc_scores(output, train_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss.update(loss.item())
            avg_acc.update(acc)

        return avg_loss.avg, avg_acc.avg
    
    def valid_step(self, valid_loader: DataLoader) -> tuple[float, float]:
        avg_loss = AvgMeter()
        avg_acc = AvgMeter()
        for valid_X, valid_y in valid_loader:        
            valid_X = valid_X.to(self.device)
            valid_y = valid_y.to(self.device)

            output = self.__model(valid_X)
            with torch.no_grad():
                loss, acc = self.calc_scores(output, valid_y)

            avg_loss.update(loss.item())
            avg_acc.update(acc)

        return avg_loss.avg, avg_acc.avg

class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.__avg = 0.0
        self.__sum = 0.0
        self.__count = 0

    def update(self, value):
        self.__count += 1
        self.__sum += value
        self.__avg = self.__sum / self.__count
    
    @property
    def avg(self):
        return self.__avg
    
class History():
    def __init__(self):
        self.__hist = []

    def update(self, value):
        self.__hist.append(value)

    @property
    def hist(self):
        return self.__hist