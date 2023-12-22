import numpy as np

from pq_amm_cnn import PQ_AMM_CNN

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


class BasicBlock_AMM():
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.relu = np.maximum

        if stride != 1 or in_channels != BasicBlock_AMM.expansion * out_channels:
            self.shortcut = True
    '''
    model.conv_bn_2d_exact/amm() are equivalent to model.Conv2d() + model.BatchNorm2d()
    '''
    # fuzed convolutional + batch norm operation
    def conv_bn_2d_exact(self, input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps=1e-5, padding=1, stride=1):
        # reshape to 4D tensors
        gamma = bn_weight.reshape(-1, 1, 1, 1)
        beta = bn_bias.reshape(-1, 1, 1, 1)
        mean = mean.reshape(-1, 1, 1, 1)
        variance = var.reshape(-1, 1, 1, 1)

        # adjust kernel weights and biases to account for batch normalization
        normalized_variance = np.sqrt(variance + eps)
        adjusted_kernel_weights = kernel_weights * gamma / normalized_variance
        adjusted_bias = beta - mean * gamma / normalized_variance

        # transform via im2col, which stretches out patches of the input image, to apply GEMM
        col_matrix, kernel_reshaped, output_shape = im2col_transform(
            input_data, adjusted_kernel_weights, adjusted_kernel_weights.shape[-1], padding, stride
        )

        # do GEMM and reshape back into an image
        conv_result = np.dot(col_matrix, kernel_reshaped.transpose())
        reshaped_output = conv_result.reshape(*output_shape).transpose(0, 3, 1, 2)

        # add bias
        res = reshaped_output + adjusted_bias.reshape(1, -1, 1, 1)

        # final_result has the shape (batch_size, num_filters, output_height, output_width)
        return res
                                                                                    
    # fuzed convolutional + batch norm operation using AMM
    def conv_bn_2d_amm(self, input_data, kernel_weights, bn_weight, bn_bias, mean, var, eps=1e-5, padding=1, stride=1, ncodebooks=1, kcentroids=64):
        # reshape to 4D tensors
        gamma = bn_weight.reshape(-1, 1, 1, 1)
        beta = bn_bias.reshape(-1, 1, 1, 1)
        mean = mean.reshape(-1, 1, 1, 1)
        variance = var.reshape(-1, 1, 1, 1)

        # adjust kernel weights and biases to account for batch normalization
        normalized_variance = np.sqrt(variance + eps)
        adjusted_kernel_weights = kernel_weights * gamma / normalized_variance
        adjusted_bias = beta - mean * gamma / normalized_variance

        # transform via im2col, which stretches out patches of the input image, to apply GEMM
        col_matrix, kernel_reshaped, output_shape = im2col_transform(
            input_data, adjusted_kernel_weights, adjusted_kernel_weights.shape[-1], padding, stride
        )
        
        # flatten image batch & tranpose kernel
        col_matrix_2d = col_matrix.reshape(-1, col_matrix.shape[-1])
        kernel_reshaped = kernel_reshaped.tranpose()

        # do AMM and reshape back into an image
        est = PQ_AMM_CNN(ncodebooks, kcentroids)
        est.fit(col_matrix_2d, kernel_reshaped)
        est.reset_for_new_task()
        conv_result = est.predict_cnn(col_matrix_2d, kernel_reshaped)
        reshaped_output = conv_result.reshape(*output_shape).tranpose(0, 3, 1, 2)

        # add bias
        res = reshaped_output + adjusted_bias.reshape(1, -1, 1, 1)

        # final_result has the shape (batch_size, num_filters, output_height, output_width)
        return res

    def residual_function(self, x):
        out = self.conv_bn_2d_amm(x)
        out = self.relu(0, out)
        out = self.conv_bn_2d_amm(out)

    def shortcut(self, x):
        if self.shortcut:
            x = self.conv_bn_2d_amm(x)
        return x

    def forward(self, x):
        return self.relu(self.residual_function(x)+self.shortcut(x))

class ResNet_AMM():
    def __init__(self, block, num_block, num_classes, num_channels, model, N_SUBSPACE, K_CLUSTER):
        self.in_planes = 

    
    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block())

    def forward():
        


def resnet18_amm(num_classes, num_channels, N_SUBSPACE, K_CLUSTER):
    return ResNet_AMM(BasicBlock_AMM, [2,2,2,2], num_classes, num_channels, N_SUBSPACE, K_CLUSTER)
        
def resnet34_amm(num_classes, num_channels, N_SUBSPACE, K_CLUSTER):
    return ResNet_AMM(BasicBlock_AMM, [3,4,6,3], num_classes, num_channels, N_SUBSPACE, K_CLUSTER)

