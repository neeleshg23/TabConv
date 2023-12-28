from models_amm.resnet_amm import resnet14_AMM, resnet18_AMM, resnet34_AMM 
from models_amm.amm.kmeans import set_gpu
from models.resnet import resnet14, resnet18, resnet34
def select_model(name):
    if name == 'r14':
        return resnet14
    elif name == 'r18':
        return resnet18 
    elif name == 'r34':
        return resnet34

def select_model_amm(name):
    if name == 'r14':
        return resnet14_AMM
    elif name == 'r18':
        return resnet18_AMM
    elif name == 'r34':
        return resnet34_AMM

def do_set_gpu(gpu):
    set_gpu(gpu)     