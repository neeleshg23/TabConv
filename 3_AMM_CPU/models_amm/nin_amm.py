import torch
import torch.nn as nn
import torch.utils.dlpack as dlpack
import numpy as np

from .amm.pq_amm_cnn import PQ_AMM_CNN

def im2col(input_data, kernel_size, stride, pad):
    if len(input_data.shape) == 2: return input_data
    N, C, H, W = input_data.shape 
    out_h = (H + 2*pad - kernel_size) // stride + 1
    out_w = (W + 2*pad - kernel_size) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    img = np.asarray(img)
    col = np.zeros((N, C, kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride*out_h
        for x in range(kernel_size):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def extract_weights_from_nin(state_dict):
    weights = {}
    biases = {}
    for i, block_key in enumerate(['block1', 'block2', 'block3']):
        for j in [0, 2, 4]:
            layer_key = f'{block_key}.{j}'
            weights[f'{layer_key}_weights'] = np.asarray(state_dict[f'{layer_key}.weight'].detach().numpy())
            biases[f'{layer_key}_bias'] = np.asarray(state_dict[f'{layer_key}.bias'].detach().numpy())
    return weights, biases

class NiNBlock_AMM:
    def __init__(self, weights, biases, ncodebook, kcentroid):
        self.weights = weights
        self.biases = biases
        self.n = ncodebook
        self.k = kcentroid


    def forward(self, x, layer_keys, switch):
        block_est = []
        intermediates = []
        for idx, layer_key in enumerate(layer_keys):
            pad = 0
            if ('block1' in layer_key or 'block2' in layer_key) and (idx == 0):
                pad = 2
            elif ('block3' in layer_key) and (idx == 0):
                pad = 1 
            if switch[idx]:
                x, est = self.conv2d_amm(x, self.weights[f'{layer_key}_weights'], self.biases[f'{layer_key}_bias'], pad=pad)
                block_est.append(est)
            else:
                x = self.conv2d(x, self.weights[f'{layer_key}_weights'], self.biases[f'{layer_key}_bias'], pad=pad)
            intermediates.append(x)
            x = self.relu(x)
        return x, block_est, intermediates
    
    def forward_eval(self, block_est, x, layer_keys, switch):
        intermediates = []
        for idx, layer_key in enumerate(layer_keys):
            pad = 0 
            if ('block1' in layer_key or 'block2' in layer_key) and (idx == 0):
                pad = 2
            elif ('block3' in layer_key) and (idx == 0):
                pad = 1 
            if switch[idx]:
                est = block_est.pop(0) 
                x = self.conv2d_eval(est, x, self.weights[f'{layer_key}_weights'], self.biases[f'{layer_key}_bias'], pad=pad)
            else:
                x = self.conv2d(x, self.weights[f'{layer_key}_weights'], self.biases[f'{layer_key}_bias'], pad=pad)
            intermediates.append(x)
            x = self.relu(x)
        return x, intermediates
    
    def conv2d(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        out = np.dot(col, col_W) + b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def conv2d_amm(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])

        est = PQ_AMM_CNN(self.n, self.k)  
        est.fit(col_matrix_2d, col_W)

        est.reset_for_new_task()
        est.set_B(col_W)
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out, est
    
    def conv2d_eval(self, est, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])
        
        est.reset_enc()
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out

    def relu(self, x):
        return np.maximum(0, x)
    
class NiN_AMM():
    def __init__(self, state_dict, ncodebook=8, kcentroid=2048):
        self.n = ncodebook
        self.k = kcentroid

        weights, biases = extract_weights_from_nin(state_dict)
        self.block1 = NiNBlock_AMM(weights, biases, ncodebook, kcentroid)
        self.block2 = NiNBlock_AMM(weights, biases, ncodebook, kcentroid)
        self.block3 = NiNBlock_AMM(weights, biases, ncodebook, kcentroid)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval()
        self.dropout = nn.Dropout(0.5).eval()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).eval()
        
        self.amm_estimators = []
        self.amm_queue = []
        
    def forward_switch(self, x, switch):
        intermediate = []
        x, est1, i1 = self.block1.forward(x, ['block1.0', 'block1.2', 'block1.4'], switch[:3])
        self.amm_estimators.append(est1)  # Save estimators
        intermediate.extend(i1)
        
        x = torch.from_dlpack(x.toDlpack())
        x = self.max_pool(x)
        x = self.dropout(x)
        x = np.from_dlpack(dlpack.to_dlpack(x))
        
        x, est2, i2 = self.block2.forward(x, ['block2.0', 'block2.2', 'block2.4'], switch[3:6])
        self.amm_estimators.append(est2)  # Save estimators
        intermediate.extend(i2)
         
        x = torch.from_dlpack(x.toDlpack())
        x = self.max_pool(x)
        x = self.dropout(x)
        x = np.from_dlpack(dlpack.to_dlpack(x))
       
        x, est3, i3 = self.block3.forward(x, ['block3.0', 'block3.2', 'block3.4'], switch[6:9])
        self.amm_estimators.append(est3)  # Save estimators
        intermediate.extend(i3)
         
        x = torch.from_dlpack(x.toDlpack())
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = np.from_dlpack(dlpack.to_dlpack(x))
        
        self.amm_queue = self.amm_estimators.copy()
        self.amm_estimators = []
        return x, intermediate


    def forward_eval(self, x, switch):
        intermediate = []
        block_est = self.amm_queue.pop(0)
        x, i1 = self.block1.forward_eval(block_est, x, ['block1.0', 'block1.2', 'block1.4'], switch[:3])
        intermediate.extend(i1)
         
        x = torch.from_dlpack(x.toDlpack())
        x = self.max_pool(x)
        x = self.dropout(x)
        x = np.from_dlpack(dlpack.to_dlpack(x))
        
        block_est = self.amm_queue.pop(0)
        x, i2 = self.block2.forward_eval(block_est, x, ['block2.0', 'block2.2', 'block2.4'], switch[3:6])
        intermediate.extend(i2)
         
        x = torch.from_dlpack(x.toDlpack())
        x = self.max_pool(x)
        x = self.dropout(x)
        x = np.from_dlpack(dlpack.to_dlpack(x))
        
        block_est = self.amm_queue.pop(0)
        x, i3 = self.block3.forward_eval(block_est, x, ['block3.0', 'block3.2', 'block3.4'], switch[6:9])
        intermediate.extend(i3)
         
        x = torch.from_dlpack(x.toDlpack())
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = np.from_dlpack(dlpack.to_dlpack(x))
        
        return x, intermediate