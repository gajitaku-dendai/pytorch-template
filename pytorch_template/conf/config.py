from dataclasses import dataclass

@dataclass
class Data():
    name: str = ""
    path: str = ""
    stride: int = 0
    patch_size: int = 0
    size: int = 0
    h: int = 0
    w: int = 0
    train: list = None
    valid: list = None
    test: list = None

@dataclass
class Model():
    name: str = ""
    pretrained_path: str = ""
    input_size: int = 0
    output_size: int = 0
    learning_rate: float = 0.0
    gamma: float = 0.0
    l2_rate: float = 0.0
    epochs: int = 0
    batch_size: int = 0
    optimizer: str = ""
    criterion: str = ""
    early: int = 0
    data_aug: bool = False
    from_rgb: bool = False

    # HSCNN-Plus
    num_blocks: int = 0

    # MST-plus-plus



@dataclass
class MyConfig:
    model: Model = Model()
    data: Data = Data()
    in_channels: int = 0
    out_channels: int = 0
    only_test: bool = True
    output_path: str = ""

