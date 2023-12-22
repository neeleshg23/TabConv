import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import amm.vq_amm as vq_amm 
from amm.pq_amm_cnn import PQ_AMM_CNN

def im2col_transform(input_data, kernel_weights, kernel_size, padding=1, stride=1):
    batch_size, input_channel, input_height, input_width = input_data.shape
    output_channel, _, kernel_height, kernel_width = kernel_weights.shape

    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    col_matrix = np.zeros((batch_size, output_height * output_width, input_channel * kernel_height * kernel_width))

    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride
            row_end = row_start + kernel_height
            col_start = j * stride
            col_end = col_start + kernel_width

            col_matrix[:, i * output_width + j, :] = padded_input[:, :, row_start:row_end, col_start:col_end].reshape(
                batch_size, -1)

    reshaped_kernel_weights = kernel_weights.reshape(output_channel, -1)
    output_shape = [batch_size, output_height, output_width, output_channel]
    return col_matrix, reshaped_kernel_weights, output_shape



def im2col_conv_bn_fold(input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps=1e-5, padding=1, stride=1):
    # https://scortex.io/batch-norm-folding-an-easy-way-to-improve-your-network-speed/
    gamma, beta = bn_weight.reshape(-1, 1, 1, 1), bn_bias.reshape(-1, 1, 1, 1)
    mean, var = mean.reshape(-1, 1, 1, 1), var.reshape(-1, 1, 1, 1)
    kernel_weights2 = kernel_weights * gamma / np.sqrt(var + eps)
    kernal_bias = beta + (-mean) * gamma / np.sqrt(var + eps)

    col_matrix, kernel_reshaped, output_shape = im2col_transform(input_data, kernel_weights2,
                                                                 kernel_weights2.shape[-1], padding, stride)

    conv_result = np.dot(col_matrix, kernel_reshaped.transpose())
    # col_matrix: (3025, 100, 9), kernel: (4,9), output,(3025, 100, 4)
    output = conv_result.reshape(*output_shape).transpose(0, 3, 1, 2)
    res = output + kernal_bias.reshape(1, -1, 1, 1)
    # output: (3025, 4, 10, 10) + (1,4,1,1)
    # res: (3025, 4, 10, 10)
    return res


def im2col_conv_bn_fold_amm(input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps=1e-5,
                            ncodebooks=1, ncentroids=64, padding=1, stride=1, est=None):
    # https://scortex.io/batch-norm-folding-an-easy-way-to-improve-your-network-speed/
    gamma, beta = bn_weight.reshape(-1, 1, 1, 1), bn_bias.reshape(-1, 1, 1, 1)
    mean, var = mean.reshape(-1, 1, 1, 1), var.reshape(-1, 1, 1, 1)
    kernel_weights2 = kernel_weights * gamma / np.sqrt(var + eps)
    kernel_bias = beta + (-mean) * gamma / np.sqrt(var + eps)

    col_matrix, kernel_reshaped, output_shape = im2col_transform(input_data, kernel_weights2,
                                                                 kernel_weights2.shape[-1], padding, stride)

    kernel_reshaped = kernel_reshaped.transpose()
    col_matrix_2d = col_matrix.reshape(-1, col_matrix.shape[-1])

    flag_return_est = False
    if est is None:
        est = PQ_AMM_CNN(ncodebooks=ncodebooks, ncentroids=ncentroids)
        est.fit(col_matrix_2d, kernel_reshaped)
        flag_return_est = True

    est.reset_for_new_task()
    est.set_B(kernel_reshaped)
    conv_result = est.predict_cnn(col_matrix_2d, kernel_reshaped)

    output = conv_result.reshape(*output_shape).transpose(0, 3, 1, 2)
    res = output + kernel_bias.reshape(1, -1, 1, 1)
    return (res, est) if flag_return_est else res


def mlp_amm(input_data, weights, bias, act_fn=None, ncodebooks=1, ncentroids=64, est=None):
    flag_return_est = False
    if est is None:
        est = vq_amm.PQMatmul(ncodebooks, ncentroids)
        est.fit(input_data, weights)
        flag_return_est = True

    est.reset_for_new_task()
    est.set_B(weights)
    res = est.predict(input_data, weights) + bias
    if act_fn is not None:
        res = act_fn(torch.tensor(res)).detach().numpy()
    return (res, est) if flag_return_est else res


class ResNet_Tiny_Manual():
    def __init__(self, model, N_SUBSPACE, K_CLUSTER, record_res_bool=True):
        # self.model=model
        self.layer1_weights = self.get_param(model.conv1)
        self.layer1_bn_buffers = self.get_param(model.conv1, 'buffer')

        self.layer2_weights = self.get_param(model.conv2_x)
        self.layer2_bn_buffers = self.get_param(model.conv2_x, 'buffer')

        self.layer3_weights = self.get_param(model.conv3_x)
        self.layer3_bn_buffers = self.get_param(model.conv3_x, 'buffer')

        self.layer4_weights = self.get_param(model.conv4_x)
        self.layer4_bn_buffers = self.get_param(model.conv4_x, 'buffer')
        
        self.layer_fc_weights = self.get_param(model.fc)

        self.relu = np.maximum  # numpy implementation of ReLU
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.n_subspace = N_SUBSPACE
        self.k_cluster = K_CLUSTER
        self.record_res_bool=record_res_bool

        self.amm_estimators = []
        self.amm_est_queue=[]

        self.mm_res = []
        
        self.fine_tune_targets = []
        

    def get_param(self, model_layer, type="parameters"):
        if type == "parameters":
            return [param.detach().numpy() for param in model_layer.parameters()]
        elif type == "buffer":  # extract BN mean and var, remove track_num in buffer
            return [param.detach().numpy() for param in model_layer.buffers() if param.numel() > 1]
        else:
            raise ValueError("Invalid type in model layer to get parameters")

    def resnet_im2col_conv_bn_fold_amm_exact(self, input_data, kernel_weights, bn_weight, bn_bias, mean, var,
                                             eps=1e-5, padding=1, stride=1):
        res = im2col_conv_bn_fold(input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps, padding, stride)
        self.mm_res.append(res)
        return res
    def resnet_im2col_conv_bn_fold_amm(self, input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps=1e-5,
                            ncodebooks=1, ncentroids=64, padding=1, stride=1, est=None):

        res = im2col_conv_bn_fold_amm(input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps,
                                    ncodebooks, ncentroids, padding, stride, est)

        if isinstance(res, tuple):
            self.mm_res.append(res[0])
            self.amm_estimators.append(res[1])
            return res[0]
        else:
            self.mm_res.append(res)
            return res

    def resnet_mlp(self, input_data, weights, bias, act_fn=None, ncodebooks=1, ncentroids=64, est=None, fine_tune=False):
        if fine_tune:
            target = self.fine_tune_targets
            weights, bias = self.fine_tune_fc_layer(torch.tensor(input_data), weights, bias, torch.tensor(target))
        res = mlp_amm(input_data, weights, bias, act_fn, ncodebooks, ncentroids, est)
        if isinstance(res, tuple):
            self.mm_res.append(res[0])
            self.amm_estimators.append(res[1])
            return res[0]
        else:
            self.mm_res.append(res)
            return res

    def forward_block_exact(self, input_data, weights, bn_buffer, stride=1):
        x = self.resnet_im2col_conv_bn_fold_amm_exact(input_data, *weights[0:3], *bn_buffer[0:2], padding=1, stride=stride)
        x = self.relu(0, x)
        x = self.resnet_im2col_conv_bn_fold_amm_exact(x, *weights[3:6], *bn_buffer[2:4])
        if stride == 1:
            x += input_data
        else:
            x += self.resnet_im2col_conv_bn_fold_amm_exact(input_data, *weights[6:9], *bn_buffer[4:6], padding=0, stride=stride)
        x = self.relu(0, x)
        return x

    def train_block_amm(self, input_data, weights, bn_buffer, stride=1, n=1, k=64):

        x = self.resnet_im2col_conv_bn_fold_amm(input_data, *weights[0:3], *bn_buffer[0:2], ncodebooks=n, ncentroids=k,
                                          padding=1, stride=stride)

        x = self.relu(0, x)
        x = self.resnet_im2col_conv_bn_fold_amm(x, *weights[3:6], *bn_buffer[2:4], ncodebooks=n, ncentroids=k,
                                          padding=1, stride=1)
        if stride == 1:
            x += input_data
        else:
            res = self.resnet_im2col_conv_bn_fold_amm(input_data, *weights[6:9], *bn_buffer[4:6], ncodebooks=1, ncentroids=k,
                                                padding=0, stride=stride)
            x += res

        x = self.relu(0, x)

        return x

    def eval_block_amm(self, input_data, weights, bn_buffer, stride=1, n=1, k=64):
        x = self.resnet_im2col_conv_bn_fold_amm(input_data, *weights[0:3], *bn_buffer[0:2], ncodebooks=n, ncentroids=k,
                                    padding=1, stride=stride, est=self.amm_est_queue.pop(0))
        x = self.relu(0, x)
        x = self.resnet_im2col_conv_bn_fold_amm(x, *weights[3:6], *bn_buffer[2:4], ncodebooks=n, ncentroids=k,
                                    padding=1, stride=1, est=self.amm_est_queue.pop(0))

        if stride == 1:
            x += input_data
        else:
            x += self.resnet_im2col_conv_bn_fold_amm(input_data, *weights[6:9], *bn_buffer[4:6], ncodebooks=1, ncentroids=k,
                                         padding=0, stride=stride, est=self.amm_est_queue.pop(0))
        x = self.relu(0, x)
        return x

    def forward_exact_bn_fold(self, input_data):
        layer1 = self.resnet_im2col_conv_bn_fold_amm_exact(input_data, *self.layer1_weights, *self.layer1_bn_buffers)
        layer1 = self.relu(0, layer1)
        layer2 = self.forward_block_exact(layer1, self.layer2_weights, self.layer2_bn_buffers, stride=1)
        layer3 = self.forward_block_exact(layer2, self.layer3_weights, self.layer3_bn_buffers, stride=2)
        layer4 = self.forward_block_exact(layer3, self.layer4_weights, self.layer4_bn_buffers, stride=2)

        output = self.avg_pool(torch.tensor(layer4)).view(input_data.shape[0], -1).detach().numpy()  # torch.Size([3025, 16, 1, 1])
        # output = output.view(output.size(0), -1) #torch.Size([3025, 16])
        output = np.dot(output, self.layer_fc_weights[0].transpose()) + self.layer_fc_weights[1]
        self.mm_res.append(output)
        output = self.sigmoid(torch.tensor(output)).detach().numpy()
        layer_res = [layer1, layer2, layer3, layer4, output]
        mm_res = self.mm_res.copy()
        self.mm_res.clear()
        return layer_res, mm_res
    
    def train_amm(self, input_data, mask, fine_tune=False):
        if mask[0]:
            layer1 = self.resnet_im2col_conv_bn_fold_amm(input_data, *self.layer1_weights, *self.layer1_bn_buffers, ncodebooks=self.n_subspace[0], ncentroids=self.k_cluster[0])
        else:
            layer1 = self.resnet_im2col_conv_bn_fold_amm_exact(input_data, *self.layer1_weights, *self.layer1_bn_buffers)
            
        layer1 = self.relu(0, layer1)

        if mask[1]:
            layer2 = self.train_block_amm(layer1, self.layer2_weights, self.layer2_bn_buffers, stride=1, n=self.n_subspace[1], k=self.k_cluster[1])
        else:
            layer2 = self.forward_block_exact(layer1, self.layer2_weights, self.layer2_bn_buffers, stride=1)
        
        if mask[2]:
            layer3 = self.train_block_amm(layer2, self.layer3_weights, self.layer3_bn_buffers, stride=2, n=self.n_subspace[2], k=self.k_cluster[2])
        else:
            layer3 = self.forward_block_exact(layer2, self.layer3_weights, self.layer3_bn_buffers, stride=2)
        
        if mask[3]:
            layer4 = self.train_block_amm(layer3, self.layer4_weights, self.layer4_bn_buffers, stride=2, n=self.n_subspace[3], k=self.k_cluster[3])
        else:
            layer4 = self.forward_block_exact(layer3, self.layer4_weights, self.layer4_bn_buffers, stride=2)
            
        output = self.avg_pool(torch.tensor(layer4)).view(input_data.shape[0], -1).detach().numpy()
        
        if mask[4]:
            output = self.resnet_mlp(output, self.layer_fc_weights[0].transpose(), self.layer_fc_weights[1], ncodebooks=self.n_subspace[4], ncentroids=self.k_cluster[4], act_fn=None, fine_tune=fine_tune)
        else:
            output = np.dot(output, self.layer_fc_weights[0].transpose()) + self.layer_fc_weights[1]
            self.mm_res.append(output)
        output = self.sigmoid(torch.tensor(output)).detach().numpy()

        self.amm_est_queue = self.amm_estimators.copy()
        layer_res = [layer1, layer2, layer3, layer4, output]
        mm_res = self.mm_res.copy()
        self.mm_res.clear()
        return layer_res, mm_res

    def eval_amm(self, input_data, mask):
        if self.amm_est_queue:
            self.amm_est_queue = self.amm_estimators.copy()
       
        if mask[0]:
            layer1 = self.resnet_im2col_conv_bn_fold_amm(input_data, *self.layer1_weights, *self.layer1_bn_buffers, ncodebooks=self.n_subspace[0], ncentroids=self.k_cluster[0], est=self.amm_est_queue.pop(0))
        else:
            layer1 = self.resnet_im2col_conv_bn_fold_amm_exact(input_data, *self.layer1_weights, *self.layer1_bn_buffers)
        layer1 = self.relu(0, layer1)

        if mask[1]:
            layer2 = self.eval_block_amm(layer1, self.layer2_weights, self.layer2_bn_buffers, stride=1, n=self.n_subspace[1], k=self.k_cluster[1])
        else:
            layer2 = self.forward_block_exact(layer1, self.layer2_weights, self.layer2_bn_buffers, stride=1)
        if mask[2]:
             layer3 = self.eval_block_amm(layer2, self.layer3_weights, self.layer3_bn_buffers, stride=2, n=self.n_subspace[2], k=self.k_cluster[2])
        else:
            layer3 = self.forward_block_exact(layer2, self.layer3_weights, self.layer3_bn_buffers, stride=2)
        if mask[3]: 
            layer4 = self.eval_block_amm(layer3, self.layer4_weights, self.layer4_bn_buffers, stride=2, n=self.n_subspace[3], k=self.k_cluster[3])
        else:
            layer4 = self.forward_block_exact(layer3, self.layer4_weights, self.layer4_bn_buffers, stride=2)
            
        output = self.avg_pool(torch.tensor(layer4)).view(input_data.shape[0], -1).detach().numpy()
        if mask[4]:
            output = self.resnet_mlp(output, self.layer_fc_weights[0].transpose(), self.layer_fc_weights[1], act_fn=None, ncodebooks=self.n_subspace[4], ncentroids=self.k_cluster[4], est=self.amm_est_queue.pop(0))
        else:
            output = np.dot(output, self.layer_fc_weights[0].transpose()) + self.layer_fc_weights[1]
            self.mm_res.append(output)
        output = self.sigmoid(torch.tensor(output)).detach().numpy()

        layer_res = [layer1, layer2, layer3, layer4, output]
        mm_res = self.mm_res.copy()
        self.mm_res.clear()
        return layer_res, mm_res
   
    def fine_tune_fc_layer(self, new_input, weight, bias, target, epoch=300, lr=0.001):
        linear_layer = nn.Linear(weight.shape[0], weight.shape[1])
        with torch.no_grad():
            linear_layer.weight.copy_(torch.tensor(weight).float().t())  # Transpose the weight matrix
            linear_layer.bias.copy_(torch.tensor(bias).float())

        new_input = new_input.float()
        target = target.float() 

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

        new_weight, new_bias = linear_layer.weight.detach().numpy(), linear_layer.bias.detach().numpy()
        return new_weight.T, new_bias  # Transpose back the weight matrix
    
    def fine_tune(self, input_data, targets, mask):
        self.fine_tune_targets = targets 
        return self.train_amm(input_data, mask, fine_tune=True)
    