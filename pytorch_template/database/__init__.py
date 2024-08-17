from pytorch_template.database.aaa import aaa_dataset
from pytorch_template.conf.config import MyConfig

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_dataset(cfg: MyConfig):
    if cfg.data.name == 'aaa':
        dataset = aaa_dataset.MyDataset(cfg)
    else:
        print(f'Dataset {cfg.data.name} is not defined !!!!')
    labels = [label for _, label, _, _ in dataset]
    indices = list(range(len(dataset)))
    # if cfg.data.onlyTrain:
    #     train_dataset = Subset(dataset,indices)
    #     train_dataset = aaa_dataset.TrainDataset(train_dataset)
    #     return train_dataset, None
    train_indices, valid_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=1)
    train_dataset = Subset(dataset,train_indices)
    train_dataset = aaa_dataset.TrainDataset(train_dataset)
    valid_dataset = Subset(dataset,valid_indices)
    valid_dataset = aaa_dataset.ValidDataset(valid_dataset)
    return train_dataset, valid_dataset