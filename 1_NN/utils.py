from models.resnet import resnet_tiny, resnet18
from models.alexnet import AlexNet

def select_model(name):
    if name == 'a':
        return AlexNet
    elif name == 'rt':
        return resnet_tiny
    elif name == 'r18':
        return resnet18
     