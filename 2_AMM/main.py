import argparse
import numpy as np
import torch
import torch.multiprocessing
from sklearn.metrics import accuracy_score

from data_loader import get_data
from utils import select_model, select_model_amm, do_set_gpu

torch.multiprocessing.set_sharing_strategy('file_system')

def split(data_loader):
    all_data, all_targets = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        all_data.append(data)
        all_targets.append(target)
    all_data_tensor = torch.cat(all_data, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    return all_data_tensor, all_targets_tensor

def get_predictions(scores):
    """Convert continuous scores to binary predictions."""
    return np.argmax(scores, axis=1)

VAL_SPLIT = 0

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', type=str, required=True, help='Model abbreviation')
parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
parser.add_argument('--gpu', '-g', type=int, required=True, help='GPU number')
parser.add_argument('--ncodebook', '-n', type=int, required=True, help='Number of subspaces per MM')
parser.add_argument('--kcentroid', '-k', type=int, required=True, help='Number of centroids per subspace')
parser.add_argument('--train', '-tr', type=int, required=True, help='Number of train images')
parser.add_argument('--test', '-te', type=int, required=True, help='Number of train images')

args = parser.parse_args()

do_set_gpu(args.gpu)
torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

dataset = args.dataset
N_TRAIN = args.train
N_TEST = args.test

train_loader, _, test_loader, num_classes, num_channels = get_data('/data/narayanan/CF', dataset, VAL_SPLIT)
train_data, train_target = split(train_loader)
test_data, test_target = split(test_loader)

train_data, train_target = train_data[:N_TRAIN], train_target[:N_TRAIN]
test_data, test_target = test_data[:N_TEST], test_target[:N_TEST]  

model = select_model(args.model)(num_classes, num_channels)
model.load_state_dict(torch.load(f'../0_RES/1_NN/{args.model}-{args.dataset}.pth', map_location=torch.device('cpu')))
model.eval()

NCODEBOOK = args.ncodebook
KCENTROID = args.kcentroid
model_amm = select_model_amm(args.model)(model.state_dict(), NCODEBOOK, KCENTROID)

train_res = model(train_data).detach().numpy()
test_res = model(test_data).detach().numpy()

''' 
# non-AMM, manual 
train_res_amm = model_amm.forward(train_data)
test_res_amm = model_amm.forward(test_data)
'''
train_res_amm = model_amm.forward_amm(train_data)
test_res_amm = model_amm.forward_amm(test_data)

train_mse = np.mean((train_res - train_res_amm)**2)
test_mse = np.mean((test_res - test_res_amm)**2)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

# get NN accuracy
train_pred = get_predictions(train_res)
test_pred = get_predictions(test_res)

train_accuracy = accuracy_score(train_target, train_pred)
test_accuracy = accuracy_score(test_target, test_pred)

print(f'NN - Train accuracy: {train_accuracy}')
print(f'NN - Test accuracy: {test_accuracy}')

# get AMM accuracy
train_pred_amm = get_predictions(train_res_amm)
test_pred_amm = get_predictions(test_res_amm)

train_accuracy_amm = accuracy_score(train_target, train_pred_amm)
test_accuracy_amm = accuracy_score(test_target, test_pred_amm)

print(f'AMM - Train accuracy: {train_accuracy_amm}')
print(f'AMM - Test accuracy: {test_accuracy_amm}')

# save these values to model-dataset-ncodebook-kcentroid-train-test.txt
path = f'../0_RES/2_AMM/{args.model}-{args.dataset}-{NCODEBOOK}-{KCENTROID}-{N_TRAIN}-{N_TEST}.log'

with open(path, 'w') as f:
    f.write(f'Train MSE: {train_mse}\n')
    f.write(f'Test MSE: {test_mse}\n')
    f.write(f'NN - Train accuracy: {train_accuracy}\n')
    f.write(f'NN - Test accuracy: {test_accuracy}\n')
    f.write(f'AMM - Train accuracy: {train_accuracy_amm}\n')
    f.write(f'AMM - Test accuracy: {test_accuracy_amm}\n')