import argparse
import torch
import torch.optim as optim
from torchinfo import summary

from logger import Logger
from data_loader import get_data
from utils import select_model
from train import run_epoch
from test import run_test

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset abbreviation')
parser.add_argument('--model', '-m', type=str, required=True, help='Model to train abbreviation')
parser.add_argument('--gpu', '-g', type=int, required=True, help='GPU number')
args = parser.parse_args()

EPOCHS = 200 
EARLY_STOP = 10 
LR = 0.00075
VAL_SPLIT = 0.2
DATASET = args.dataset
model_abbrev = args.model 

model_save_path = f'../0_RES/1_NN/{model_abbrev}-{DATASET}'

log = Logger()
log.set_logger(f'{model_save_path}.log') 
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(args.dataset)
train_loader, val_loader, test_loader, num_classes, num_channels = get_data('/data/CV_Datasets', DATASET, VAL_SPLIT)
print(f"Number of classes: {num_classes}")
print(f"Number of channels: {num_channels}")
model = select_model(model_abbrev)(num_classes, num_channels).to(device) 
optimizer = optim.Adam(model.parameters(), LR)

run_epoch(model, train_loader, val_loader, optimizer, EPOCHS, EARLY_STOP, f'{model_save_path}.pth', device, log)
accuracy, p, r, f1 = run_test(model, test_loader, device, f'{model_save_path}.pth') 
log.logger.info("Accuracy: {:.10f}".format(accuracy))
log.logger.info("Precision: {:.10f}, Recall: {:.10f}, F1: {:.10f}".format(p, r, f1))
