import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn

def get_model(pretrained=False, num_classes=10, path=None):
    model = models.resnet18()
    if pretrained:
        if path is None:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if not pretrained:
        model.apply(init_parameters)
    if path is not None:
        weights=torch.load(path)
        model.load_state_dict(weights)

    return model

def init_parameters(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()