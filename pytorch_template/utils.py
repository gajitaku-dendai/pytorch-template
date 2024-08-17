import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import torch.optim as optim
import torch.nn as nn
import os
import yaml
import datetime
import csv
from pathlib import Path
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pytorch_template.database import get_dataset
from pytorch_template.conf.config import MyConfig

class DataProcesser:
    """DataProcesserクラス:データの取得，加工など．
    """
    def __init__(self, cfg: MyConfig):
        """_summary_

        Args:
            cfg (Config): Configクラス
        """
        self.cfg = cfg
        self.X_train_max = None
        self.X_train_min = None

    def get_train_valid_data(self):
        train_dataset, valid_dataset = get_dataset(self.cfg)
        return train_dataset, valid_dataset
    
    def dataset_to_dataloader(self, dataset, batch_size, isValid=False, isTest=False):
        if isValid or isTest:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        else:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

def get_criterion(cfg: MyConfig) -> nn.Module:
    if cfg.model.criterion == "CrossEntropy":
        return nn.CrossEntropyLoss()
    
def get_optimizer(cfg: MyConfig, model: nn.Module) -> optim.optimizer:
    if cfg.model.optimizer == "Adam":
        return optim.Adam(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)
    
    elif cfg.model.optimizer == "SGD":
        return optim.SGD(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, path, patience=5, verbose=False):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        # self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score
            self.checkpoint(score, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
                print(f"the best of val_loss: {self.best_score:.5f}")
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.checkpoint(score, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, score, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.best_score:.5f} --> {score:.5f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.best_score = score #その時のlossを記録する

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, true):
        return accuracy_score(true, pred)