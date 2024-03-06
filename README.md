# TabConv: Low-Computation CNN Inference via Table Lookups 

## Abstract

Convolutional Neural Networks (CNNs) have demonstrated remarkable ability throughout the field of computer vision. 
However, CNN inference requires a large number of arithmetic operations, making them expensive to deploy in hardware.
Current approaches alleviate this issue by developing hardware-supported, algorithmic processes to simplify spatial convolution functions. 
However, these methods still heavily rely on matrix multiplication, leading to significant computational overhead.
To bridge the gap between hardware, algorithmic acceleration, and approximate matrix multiplication, we propose \textit{TabConv}, a novel, table-based approximation for convolution to significantly reduce arithmetic operations during inference.
Additionally, we introduce a priority masking technique based on cosine similarity to select layers for table-based approximation, thereby maintaining the model performance.
We evaluate our approach on popular CNNs: ResNet-18, ResNet-34, and NetworkInNetwork (NIN).
TabConv preserves over 93\% of the original model's performance while reducing arithmetic operations by 36.5\%, 25.8\%, and 99.4\% for ResNet-18 on CIFAR-10, CIFAR-100, and MNIST, respectively, 35.6\% and 99.3\% for ResNet-34 on CIFAR-10 and MNIST, and 98.9\% for NIN on MNIST, achieving low-computation inference.
