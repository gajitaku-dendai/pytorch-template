# 必要なモジュールをimport
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats
import glob, os
from tqdm import tqdm
import wandb


# 自作クラス
from pytorch_template.utils import *
from pytorch_template.trainer import *
from pytorch_template.conf.config import MyConfig

def train(cfg: MyConfig,
          device: torch.device,
          model: nn.Module,
          optimizer: optim.optimizer,
          criterion: nn.Module,
          train_loader: DataLoader,
          valid_loader: DataLoader) -> nn.Module:
    
    earlyStopping = EarlyStopping(path=f"{cfg.output_path}/best.pth", patience=cfg.model.early, verbose=True)

    trainer = Trainer(device, model, criterion, optimizer, cfg)

    hist_loss_train = History()
    hist_acc_train = History()
    hist_loss_valid = History()
    hist_acc_valid = History()
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.model.gamma)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg.model.learning_rate * 0.1)
    for epoch in tqdm(range(cfg.model.epochs), desc="Epochs", leave=True):
        trainer.model.train()
        train_loss, train_acc = trainer.train_step(train_loader)
        hist_loss_train.update(train_loss)
        hist_acc_train.update(train_acc)

        trainer.model.eval()
        if valid_loader is not None:
            valid_loss, valid_acc = trainer.valid_step(valid_loader)
        else:
            valid_loss, valid_acc, s = trainer.valid_step_auth()
        hist_loss_valid.update(valid_loss)
        hist_acc_valid.update(valid_acc)        

        print()
        print()
        print(f"train_loss: {train_loss:.5f}, "\
              f"train_acc: {train_acc:.5f}, "\
              f"valid_loss: {valid_loss:.5f}, "\
              f"valid_acc: {valid_acc:.5f}")

        earlyStopping(valid_acc, trainer.model)

        wandb.log({"train_acc": train_acc,
                   "train_loss": train_loss,
                   "valid_acc": valid_acc,
                   "valid_loss": valid_loss,
                   "lr": optimizer.param_groups[0]['lr']})
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{cfg.output_path}/last.pth")
            valid_acc_avg = np.mean(hist_acc_valid.hist[-10:])
            wandb.log({"valid_acc_avg": valid_acc_avg})
        
        if earlyStopping.early_stop:
            print("Early Stopping!")
            break

        # scheduler.step()

    print("Finished Training")

    return trainer.model