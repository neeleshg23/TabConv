import torch.nn as nn
import torch.nn.functional as F
import cupy as cp

class NiN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(NiN, self).__init__()
        # NiN Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(num_channels, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )
        # NiN Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )
        # NiN Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        intermediate = []
        # output block 1 until each of the convolutions, append to intermediate
        x = self.block1[:1](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block1[1:3](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy())) 
        x = self.block1[3:5](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block1[5:](x)
        
        x = self.block2[:1](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block2[1:3](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block2[3:5](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block2[5:](x)
        
        x = self.block3[:1](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block3[1:3](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block3[3:5](x)
        intermediate.append(cp.asarray(x.clone().detach().numpy()))
        x = self.block3[5:](x)
       
        x = x.view(x.size(0), -1)  
        
        return x, intermediate