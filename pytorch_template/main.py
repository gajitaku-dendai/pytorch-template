import os
# CPU使用スレッド数7
os.environ["OMP_NUM_THREADS"] = "7"

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn as nn
import random
import wandb
import yaml

# 自作クラス
from pytorch_template.conf.config import MyConfig
from pytorch_template.utils import *
from pytorch_template.train import train
from pytorch_template.architecture import get_model

# GPU,CPUの定義
GPU = torch.device("cuda")
CPU = torch.device("cpu")

# GPUが使用可能ならGPU計算
device = None
if torch.cuda.is_available():
    device = GPU
else:
    device = CPU

# *.yaml の 読み込み
def load_config(file):
    with open(file, 'r') as yml:
        config = yaml.safe_load(yml)
    return config

class DictDotNotation(dict):
    """ 辞書型をドットアクセス可能にするクラス

    Args:
        dict (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictDotNotation(value)
        self.__dict__ = self
        
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DictDotNotation' object has no attribute '{key}'")

def sweep(id: str = ""):
    """wandb用ハイパラチューニング自動化機能

    Args:
        id (str, optional): sweep_id. Defaults to "".
    """
    sweep_config = load_config("sweep.yaml")
    if id == "":
        sweep_id = wandb.sweep(sweep_config, project="sample_project")
    else:
        sweep_id = id
    wandb.agent(sweep_id, main)

@hydra.main(config_name="config",version_base=None,config_path="conf")
def main(cfg: MyConfig) -> None:

    if cfg.wandb:
        # hydra用のconfigをwandbへ書き込み
        wandb.init(
            # set the wandb project where this run will be logged
            project="sample_project",

            # track hyperparameters and run metadata
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )

        # wandbのconfigは辞書型なので，ドットアクセス可能に
        cfg = DictDotNotation(wandb.config)

    # configの中身表示
    print("####################")
    print("Config")
    print("====================")
    print(cfg)
    print("####################")
    print()
    print()

    # random性排除
    torch_fix_seed()

    # outputフォルダの取得
    cfg.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if not cfg.only_test:
        print("####################")
        print("Train-Mode")
        print("####################")
        print()
        print()

        # データのロード
        print("loading data...")
        dataProcesser = DataProcesser(cfg)
        train_dataset, valid_dataset = dataProcesser.get_train_valid_data()
        labels = [label for _, label in train_dataset]
        u, counts = np.unique(labels, return_counts=True)
        print("train_details:", u, counts)
        labels = [label for _, label in valid_dataset]
        u, counts = np.unique(labels, return_counts=True)
        print("valid_details:", u, counts)
        train_loader = dataProcesser.dataset_to_dataloader(train_dataset, batch_size=cfg.model.batch_size)
        if valid_dataset is not None:
            valid_loader = dataProcesser.dataset_to_dataloader(valid_dataset, batch_size=50, isValid=True)
        else:
            valid_loader = None
        print("success to load data!")

        print()
        print()

        # モデルのロード
        print("loading model...")
        model = get_model(cfg, pretrained_flag=False).to(device)
        optimizer = get_optimizer(cfg, model)
        criterion = get_criterion(cfg)
    
        print(model)
        print(f"success to load {cfg.model.name}")
        print()
        print()
        print("starting to train...")
        model = train(cfg, device, model, optimizer, criterion, train_loader, valid_loader)

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

if __name__ == "__main__":
    main()
    # sweep()
    # sweep("033lab-bio/miru2024_supcon/1g5d2ze6")
