import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    
    def __init__(self, in_chan, out_chan, downsample):
        
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias = False),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d()
        )
        self.dwnsamp = downsample
        self.relu = nn.ReLU()
        self.out_chan = out_chan
        
    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.dwnsamp: res = self.dwnsamp(x)
        
        out += res
        out = self.relu(out)
        
        return out
    
class ResNetSingleChan(nn.Module):
    
    def __init__(self) :
        super().__init__()
        self.inplanes = 64
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        se
                
        return x
