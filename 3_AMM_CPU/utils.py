from models_amm.resnet_amm import resnet14_AMM, resnet18_AMM, resnet34_AMM 
from models_amm.nin_amm import NiN_AMM 
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

def select_model_amm(name):
    if name == 'r14':
        return resnet14_AMM
    elif name == 'r18':
        return resnet18_AMM
    elif name == 'r34':
        return resnet34_AMM
    elif name == 'n':
        return NiN_AMM 

