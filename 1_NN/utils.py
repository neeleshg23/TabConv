from models.resnet import resnet14, resnet18, resnet34
from models.nin import NiN 
def select_model(name):
    if name == 'r14':
        return resnet14
    elif name == 'r18':
        return resnet18 
    elif name == 'r34':
        return resnet34
    elif name == 'n':
        return NiN 