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

## Installation
First, edit line 33 of `1_NN/main.py` to point to the data directory
```bash
git clone www.github.com/neeleshg23/TabConv.git
cd TabConv
conda env create -f env.yaml python=3.9
conda activate rapid
mkdir -p 0_RES/1_NN; mkdir -p 0_RES/2_AMM
```

## Experiment Workflow
### Dataset Command Abbreviations
| Dataset Name | Abbreviation |
|--------------|--------------|
| CIFAR-10     | c10          |
| CIFAR-100    | c100         |
| MNIST        | m            |

### Model Command Abbreviations
| Model Name       | Abbreviation | Number of Switches |
|------------------|--------------|--------------------|
| ResNet-18        | r18          | 10                 |
| ResNet-34        | r34          | 19                 |
| NetworkInNetwork | n            | 9                  |

### Training Neural Network
```bash
cd 1_NN
python main.py -h
usage: main.py [-h] --dataset DATASET --model MODEL --gpu GPU

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Dataset abbreviation
  --model MODEL, -m MODEL
                        Model to train abbreviation
  --gpu GPU, -g GPU     GPU number
```

