import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.dlpack as dlpack
from tqdm import tqdm
import cupy as cp
import numpy as np

from .amm.pq_amm_cnn import PQ_AMM_CNN
from .amm.vq_amm import PQMatmul
from .amm.kmeans import get_device


def im2col(input_data, kernel_size, stride, pad):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - kernel_size) // stride + 1
    out_w = (W + 2*pad - kernel_size) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    img = cp.asarray(img)
    col = cp.zeros((N, C, kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride*out_h
        for x in range(kernel_size):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def extract_weights_from_basicblocks(state_dict, layer_name, num_blocks):
    weights = []
    for i in range(num_blocks):
        block_key = f'{layer_name}.{i}'

        conv1_weights = cp.asarray(state_dict[f'{block_key}.residual_function.0.weight'].detach().numpy())
        bn1_weights = cp.asarray(state_dict[f'{block_key}.residual_function.1.weight'].detach().numpy())
        bn1_bias = cp.asarray(state_dict[f'{block_key}.residual_function.1.bias'].detach().numpy())

        conv2_weights = cp.asarray(state_dict[f'{block_key}.residual_function.3.weight'].detach().numpy())
        bn2_weights = cp.asarray(state_dict[f'{block_key}.residual_function.4.weight'].detach().numpy())
        bn2_bias = cp.asarray(state_dict[f'{block_key}.residual_function.4.bias'].detach().numpy())

        if f'{block_key}.shortcut.0.weight' in state_dict:
            shortcut_conv_weights = cp.asarray(state_dict[f'{block_key}.shortcut.0.weight'].detach().numpy())
            shortcut_bn_weights = cp.asarray(state_dict[f'{block_key}.shortcut.1.weight'].detach().numpy())
            shortcut_bn_bias = cp.asarray(state_dict[f'{block_key}.shortcut.1.bias'].detach().numpy())
        else:
            shortcut_conv_weights = shortcut_bn_weights = shortcut_bn_bias = None
        
        w = (conv1_weights, bn1_weights, bn1_bias, conv2_weights, bn2_weights, bn2_bias, shortcut_conv_weights, shortcut_bn_weights, shortcut_bn_bias)
        weights.append(w)

    return weights

def extract_means_and_vars_from_basicblocks(state_dict, layer_name, num_blocks):
    means_vars = []
    for i in range(num_blocks):
        block_key = f'{layer_name}.{i}'
        
        bn1_mean = cp.asarray(state_dict[f'{block_key}.residual_function.1.running_mean'].detach().numpy())
        bn1_var = cp.asarray(state_dict[f'{block_key}.residual_function.1.running_var'].detach().numpy())
        bn2_mean = cp.asarray(state_dict[f'{block_key}.residual_function.4.running_mean'].detach().numpy())
        bn2_var = cp.asarray(state_dict[f'{block_key}.residual_function.4.running_var'].detach().numpy())
        
        if f'{block_key}.shortcut.1.running_mean' in state_dict:
            shortcut_bn_mean = cp.asarray(state_dict[f'{block_key}.shortcut.1.running_mean'].detach().numpy())
            shortcut_bn_var = cp.asarray(state_dict[f'{block_key}.shortcut.1.running_var'].detach().numpy())
        else: 
            shortcut_bn_mean = shortcut_bn_var = None
        mv = (bn1_mean, bn1_var, bn2_mean, bn2_var, shortcut_bn_mean, shortcut_bn_var)
        means_vars.append(mv)
    return means_vars


class BasicBlock_AMM:
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, ncodebooks=8, kcentroids=2048, weights=None, means_vars=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.n = ncodebooks
        self.k = kcentroids
        
        self.weights = weights if weights is not None else {}
        self.mean_vars = means_vars if means_vars is not None else {}
        
        self.conv1_weights, self.bn1_weights, self.bn1_bias, self.conv2_weights, self.bn2_weights, self.bn2_bias, self.shortcut_conv_weights, self.shortcut_bn_weights, self.shortcut_bn_bias = weights
        self.bn1_mean, self.bn1_var, self.bn2_mean, self.bn2_var, self.shortcut_bn_mean, self.shortcut_bn_var = means_vars
         
    def conv2d(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        out = cp.dot(col, col_W) + b
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

    def batch_norm(self, x, gamma, beta, moving_mean, moving_var, eps=1e-5):
        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        out = (x_flat - moving_mean) / cp.sqrt(moving_var + eps)
        out = out * gamma + beta
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out

    def relu(self, x):
        return cp.maximum(0, x)
    
    def forward(self, x):
        
        residual = x  # keep the original input for the shortcut

        out = self.conv2d(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), self.stride, pad=1)
        out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var) 
        out = self.relu(out)
        out = self.conv2d(out, self.conv2_weights, cp.zeros(self.conv2_weights.shape[0]), 1, pad=1)
        out = self.batch_norm(out, self.bn2_weights, self.bn2_bias, self.bn2_mean, self.bn2_var)
       
        if self.stride != 1 or self.in_channels != self.out_channels * self.expansion: 
            shortcut = self.conv2d(residual, self.shortcut_conv_weights, cp.zeros(self.shortcut_conv_weights.shape[0]), self.stride, pad=0)
            shortcut = self.batch_norm(shortcut, self.shortcut_bn_weights, self.shortcut_bn_bias, self.shortcut_bn_mean, self.shortcut_bn_var)
        else:
            shortcut = residual
        out += shortcut
        out = self.relu(out)
        return out
    
    def forward_amm(self, x):
        block_est = [] 
        residual = x  # keep the original input for the shortcut

        out, est = self.conv2d_amm(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), self.stride, pad=1)
        block_est.append(est)
        out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var) 
        out = self.relu(out)
        out, est = self.conv2d_amm(out, self.conv2_weights, cp.zeros(self.conv2_weights.shape[0]), 1, pad=1)
        block_est.append(est)
        out = self.batch_norm(out, self.bn2_weights, self.bn2_bias, self.bn2_mean, self.bn2_var)
       
        if self.stride != 1 or self.in_channels != self.out_channels * self.expansion: 
            shortcut, est = self.conv2d_amm(residual, self.shortcut_conv_weights, cp.zeros(self.shortcut_conv_weights.shape[0]), self.stride, pad=0)
            block_est.append(est)
            shortcut = self.batch_norm(shortcut, self.shortcut_bn_weights, self.shortcut_bn_bias, self.shortcut_bn_mean, self.shortcut_bn_var)
        else:
            shortcut = residual
        out += shortcut
        out = self.relu(out)
        return out, block_est
    
    def forward_eval(self, block_est, x):
        residual = x
        est = block_est.pop(0)
        out = self.conv2d_eval(est, x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), self.stride, pad=1)
        out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
        out = self.relu(out)
        
        est = block_est.pop(0)
        out = self.conv2d_eval(est, out, self.conv2_weights, cp.zeros(self.conv2_weights.shape[0]), 1, pad=1)
        out = self.batch_norm(out, self.bn2_weights, self.bn2_bias, self.bn2_mean, self.bn2_var)

        if self.stride != 1 or self.in_channels != self.out_channels * self.expansion:
            est = block_est.pop(0)
            shortcut = self.conv2d_eval(est, residual, self.shortcut_conv_weights, cp.zeros(self.shortcut_conv_weights.shape[0]), self.stride, pad=0)
            shortcut = self.batch_norm(shortcut, self.shortcut_bn_weights, self.shortcut_bn_bias, self.shortcut_bn_mean, self.shortcut_bn_var)
        else:
            shortcut = residual
            
        out += shortcut
        out = self.relu(out)
        return out

DIM = 64

class ResNet_AMM:
    def __init__(self, block, num_blocks, state_dict, ncodebook=8, kcentroid=2048):
        self.in_channels = DIM

        self.n = ncodebook
        self.k = kcentroid
         
        self.state_dict = state_dict
        # Assuming state_dict is a dictionary from PyTorch model's state_dict, containing all weights and biases
        self.conv1_weights = state_dict['conv1.weight'].numpy()
        self.bn1_weights = state_dict['bn1.weight'].numpy()
        self.bn1_bias = state_dict['bn1.bias'].numpy()
        self.bn1_mean = state_dict['bn1.running_mean'].numpy()
        self.bn1_var = state_dict['bn1.running_var'].numpy()

        self.layer1 = self._make_layer(block, DIM, 'layer1', num_blocks[0])
        self.layer2 = self._make_layer(block, DIM*2, 'layer2', num_blocks[1])
        self.layer3 = self._make_layer(block, DIM*4, 'layer3', num_blocks[2])
        self.layer4 = self._make_layer(block, DIM*8, 'layer4', num_blocks[3])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_weights = cp.asarray(state_dict['linear.weight'].numpy())
        self.fc_bias = cp.asarray(state_dict['linear.bias'].numpy())
        
        self.amm_estimators = []
        self.amm_queue = []
        
    def _make_layer(self, block, out_channels, name, num_blocks):
        layers = []
        strides = [2] + (num_blocks-1)*[1]
        all_weights = extract_weights_from_basicblocks(self.state_dict, name, num_blocks)
        all_means_vars = extract_means_and_vars_from_basicblocks(self.state_dict, name, num_blocks) 
        for (s, w, mv) in zip(strides, all_weights, all_means_vars):
            layers.append(block(self.in_channels, out_channels, stride=s, ncodebooks=self.n, kcentroids=self.k, weights=w, means_vars=mv))
            self.in_channels = out_channels * block.expansion
        return layers

    def conv2d(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T
        
        col_W = cp.asarray(col_W) 
        out = cp.dot(col, col_W) + b
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
        self.amm_estimators.append(est)
        
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out
    
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

    def batch_norm(self, x, gamma, beta, moving_mean, moving_var, eps=1e-5):
        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        
        moving_mean = cp.asarray(moving_mean)
        moving_var = cp.asarray(moving_var)
        gamma = cp.asarray(gamma)
        beta = cp.asarray(beta)
        
        out = (x_flat - moving_mean) / cp.sqrt(moving_var + eps)
        out = out * gamma + beta
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out

    def relu(self, x):
        return cp.maximum(0, x)
    
    def linear_amm(self, input_data, weights, bias, target):
        input_data = torch.from_dlpack(input_data.toDlpack()).float()
        target = torch.from_dlpack(target.toDlpack()).float()
        weights, bias = self.fine_tune_fc_layer(input_data, weights, bias, target, epoch=300, lr=0.001)
        est = PQMatmul(self.n, self.k)
        est.fit(input_data, weights)
        est.reset_for_new_task()
        est.set_B(weights)
        res = est.predict(input_data, weights) + bias
        self.amm_estimators.append(est)
        return res
    
    def linear_eval(self, est, input_data, weights, bias):
        est.reset_enc()
        weights, bias = cp.asarray(weights), cp.asarray(bias) 
        res = est.predict(input_data, weights) + bias
        return res
    
    # def forward(self, x):
    #     intermediate = [] 
    #     out = self.conv2d(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
    #     intermediate.append(out)
    #     out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
    #     out = self.relu(out)

    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    #     for layer_group in layers:
    #         for layer in layer_group:
    #             out = layer.forward(out)
    #             intermediate.append(out)

    #     out = torch.from_numpy(cp.asnumpy(out))
    #     out = self.adaptive_pool(out)
    #     out = out.reshape(out.shape[0], -1)
    #     out = cp.asarray(out.detach().numpy())
    #     out = cp.dot(out, self.fc_weights.T) + self.fc_bias
    #     intermediate.append(out)
    #     return out, intermediate

    # def forward_amm(self, x, target):
    #     intermediate = [] 
    #     out = self.conv2d_amm(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
    #     intermediate.append(out)
    #     out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
    #     out = self.relu(out)

    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    #     for layer_group in layers:
    #         for layer in layer_group:
    #             out = layer.forward_amm(out)
    #             intermediate.append(out)

    #     out = torch.from_numpy(cp.asnumpy(out))
    #     out = self.adaptive_pool(out)
    #     out = out.reshape(out.shape[0], -1)
    #     out = cp.asarray(out.detach().numpy())
    #     out = self.linear_amm(out, self.fc_weights.T, self.fc_bias, target)
    #     intermediate.append(out)
    #     return out, intermediate 
    
    # def forward_eval(self, x):
    #     est = self.amm_queue.pop(0)
    #     out = self.conv2d_eval(est, x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
    #     out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
    #     out = self.relu(out)
        
    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    #     for layer_group in layers:
    #         for layer in layer_group:
    #             block_ests = self.amm_queue.pop(0)
    #             out = layer.forward_eval(block_ests, out)

    #     out = torch.from_numpy(cp.asnumpy(out))
    #     out = self.adaptive_pool(out)
    #     out = out.reshape(out.shape[0], -1)
    #     out = cp.asarray(out.detach().numpy())
    #     est = self.amm_queue.pop(0)
    #     out = self.linear_eval(est, out, self.fc_weights.T, self.fc_bias)
    #     return out  
    
    def forward_switch(self, x, switch, target):
        intermediate = []
        out = self.conv2d_amm(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1) if switch[0] \
            else self.conv2d(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
        intermediate.append(out)
        out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
        out = self.relu(out)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer_group in enumerate(layers):
            for j, layer in enumerate(layer_group):
                idx = i*len(layer_group) + j + 1
                if switch[idx]:
                    out, block_est = layer.forward_amm(out)
                    self.amm_estimators.append(block_est)
                else:
                    out = layer.forward(out)
                intermediate.append(out)

        out = torch.from_dlpack(out.toDlpack())
        out = self.adaptive_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = cp.fromDlpack(dlpack.to_dlpack(out))
        
        out = self.linear_amm(out, self.fc_weights.T, self.fc_bias, target) if switch[-1] else (cp.dot(out, self.fc_weights.T) + self.fc_bias)
        intermediate.append(out)
        
        self.amm_queue = self.amm_estimators.copy()
        self.amm_estimators = []
        return out, intermediate
    
    def forward_eval(self, x, switch):
        intermediate = []
        if switch[0]:
            est = self.amm_queue.pop(0)
            out = self.conv2d_eval(est, x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
        else:
            out = self.conv2d(x, self.conv1_weights, cp.zeros(self.conv1_weights.shape[0]), stride=2, pad=1)
        intermediate.append(out)
        
        out = self.batch_norm(out, self.bn1_weights, self.bn1_bias, self.bn1_mean, self.bn1_var)
        out = self.relu(out)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer_group in enumerate(layers):
            for j, layer in enumerate(layer_group):
                if switch[i*len(layer_group) + j + 1]:
                    block_ests = self.amm_queue.pop(0)
                    out = layer.forward_eval(block_ests, out)
                else:
                    out = layer.forward(out)
                intermediate.append(out)

        out = torch.from_dlpack(out.toDlpack())
        out = self.adaptive_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = cp.fromDlpack(dlpack.to_dlpack(out))
        
        if switch[-1]:
            est = self.amm_queue.pop(0)
            out = self.linear_eval(est, out, self.fc_weights.T, self.fc_bias)
        else:
            out = cp.dot(out, self.fc_weights.T) + self.fc_bias
        intermediate.append(out)
        
        return out, intermediate

    
    def fine_tune_fc_layer(self, new_input, weight, bias, target, epoch=300, lr=0.001):
        device = get_device()
        linear_layer = nn.Linear(weight.shape[0], weight.shape[1]).to(device)
       
        weight_torch = torch.from_dlpack(weight.toDlpack()).float().to(device).t()  # also transpose the weight matrix
        bias_torch = torch.from_dlpack(bias.toDlpack()).float().to(device)
        
        with torch.no_grad():
            linear_layer.weight.copy_(weight_torch)  
            linear_layer.bias.copy_(bias_torch)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(linear_layer.parameters(), lr=lr)

        for i in tqdm(range(epoch)):
            optimizer.zero_grad()
            new_output = linear_layer(new_input)
            loss = criterion(new_output, target)
            loss.backward()
            optimizer.step()
            if loss.item() < 1e-5:
                break
        
        # use dlpack to convert to cupy array
        new_weight, new_bias = cp.fromDlpack(dlpack.to_dlpack(linear_layer.weight)), cp.fromDlpack(dlpack.to_dlpack(linear_layer.bias))
        return new_weight.T, new_bias  # Transpose back the weight matrix

def resnet14_AMM(state_dict, n, k):
    return ResNet_AMM(BasicBlock_AMM, [1, 1, 1, 1], state_dict, n, k)

def resnet18_AMM(state_dict, n, k):
    return ResNet_AMM(BasicBlock_AMM, [2, 2, 2, 2], state_dict, n, k)

def resnet34_AMM(state_dict, n, k):
    return ResNet_AMM(BasicBlock_AMM, [3, 4, 6, 3], state_dict, n, k)