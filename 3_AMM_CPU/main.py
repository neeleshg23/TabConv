#%%
import argparse
import numpy as np
import torch
import torch.multiprocessing
from sklearn.metrics import accuracy_score

from data_loader import get_data
from utils import select_model, select_model_amm 
from metrics import layer_cossim

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

def calculate_and_save_metrics(train_cossim, test_cossim, train_res, train_res_amm, test_res, test_res_amm, train_target, test_target, model, dataset, NCODEBOOK, KCENTROID, N_TRAIN, N_TEST, switch):
    print(f'-- Train cosine similarity --')
    for i, cosim in enumerate(train_cossim):
        print(f'Layer {i}: {cosim:.4f}')
    print(f'-- Test cosine similarity --')
    for i, cosim in enumerate(test_cossim):
        print(f'Layer {i}: {cosim:.4f}')
    train_mse = ((train_res - train_res_amm)**2).mean()
    test_mse = ((test_res - test_res_amm)**2).mean()

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
    switch_str = ''.join([str(x) for x in switch])
    path = f'../0_RES/2_AMM/{model}-{dataset}-{NCODEBOOK}-{KCENTROID}-{N_TRAIN}-{N_TEST}-{switch_str}.log'
    
    # save these values to model-dataset-ncodebook-kcentroid-train-test-percent.log
    # path = f'../0_RES/2_AMM/{model}-{dataset}-{NCODEBOOK}-{KCENTROID}-{N_TRAIN}-{N_TEST}-{percent}.log'

    with open(path, 'w') as f:
        # f.write(f'Percent masked: {percent}\n')
        f.write(f'Switch: {switch}\n')
        f.write(f'Train cosine similarity: {train_cossim}\n')
        f.write(f'Test cosine similarity: {test_cossim}\n')
        f.write(f'Train MSE: {train_mse}\n')
        f.write(f'Test MSE: {test_mse}\n')
        f.write(f'NN - Train accuracy: {train_accuracy}\n')
        f.write(f'NN - Test accuracy: {test_accuracy}\n')
        f.write(f'AMM - Train accuracy: {train_accuracy_amm}\n')
        f.write(f'AMM - Test accuracy: {test_accuracy_amm}\n')

VAL_SPLIT = 0

def run_experiment_mask(model, dataset, ncodebook, kcentroid, n_train, n_test, switch): 
    torch.manual_seed(0)

    train_loader, _, test_loader, num_classes, num_channels = get_data('/data/CV_Datasets', dataset, VAL_SPLIT)
    train_data, train_target = split(train_loader)
    test_data, test_target = split(test_loader)

    train_data, train_target = train_data[:n_train], train_target[:n_train]
    test_data, test_target = test_data[:n_test], test_target[:n_test]  

    base_model = select_model(model)(num_classes, num_channels)
    base_model.load_state_dict(torch.load(f'../0_RES/1_NN/{model}-{dataset}.pth', map_location=torch.device('cpu')))
    base_model.eval()

    model_amm = select_model_amm(model)(base_model.state_dict(), ncodebook, kcentroid)
    
    exact_res_train, intermediate_train_res = base_model(train_data)
    exact_res_test, intermediate_test_res = base_model(test_data)
    
    print("-- Starting Training -- ") 
    if model == 'n': 
        train_res_amm, intermediate_train_res_amm = model_amm.forward_switch(train_data, switch) 
    else:
        train_res_amm, intermediate_train_res_amm = model_amm.forward_switch(train_data, switch, np.asarray(exact_res_train.detach().numpy()))
    
    print("-- Starting Evaluation -- ")
    test_res_amm, intermediate_test_res_amm = model_amm.forward_eval(test_data, switch)

    train_cosim = layer_cossim(intermediate_train_res, intermediate_train_res_amm)
    test_cosim = layer_cossim(intermediate_test_res, intermediate_test_res_amm)
    
    calculate_and_save_metrics(train_cosim, test_cosim, exact_res_train.detach().numpy(), train_res_amm, exact_res_test.detach().numpy(), test_res_amm, train_target, test_target, model, dataset, ncodebook, kcentroid, n_train, n_test, switch)

    return intermediate_train_res
#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Model abbreviation')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--ncodebook', '-n', type=int, required=True, help='Number of subspaces per MM')
    parser.add_argument('--kcentroid', '-k', type=int, required=True, help='Number of centroids per subspace')
    parser.add_argument('--train', '-tr', type=int, required=True, help='Number of train images')
    parser.add_argument('--test', '-te', type=int, required=True, help='Number of train images')
    parser.add_argument('--switch', '-s', type=str, required=True, help='Switch')
    args = parser.parse_args()
    
    # split the switch string into a list of ints
    switch = [int(x) for x in args.switch.split(',')]

    # run the experiment
    run_experiment_mask(args.model, args.dataset, args.ncodebook, args.kcentroid, args.train, args.test, switch)
    
if __name__ == "__main__":
    main()
# %%
# run_experiment_mask('r18', 'c10', 0, 1, 16, 100, 100, [1,1,1,1,1,1,1,1,1,1])
