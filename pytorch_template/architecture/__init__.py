from pytorch_template.conf.config import MyConfig
from pytorch_template.architecture import resnet18

def get_model(cfg: MyConfig, pretrained_flag: bool=False):
    if cfg.model.name == 'resnet18':
        model = resnet18.get_model(pretrained=pretrained_flag,
                                        path=cfg.model.pretrained_path)
    else:
        print(f'Method {cfg.model.name} is not defined !!!!')
    return model