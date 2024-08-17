from torch.utils.data import Dataset, Subset
from pytorch_template.conf.config import MyConfig

class MyDataset(Dataset):
    def __init__(self, cfg: MyConfig):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
    
class TrainDataset(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset: MyDataset = subset.dataset
        self.length = len(subset)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

class ValidDataset(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset: MyDataset = subset.dataset
        self.length = len(subset)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
