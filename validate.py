
import numpy as np
import torch

def Validate(model, testloader, device):
    
    correct = 0
    total = 0
    running_acc = 0
    
    model = model.to(device)
    
    for index, data in enumerate(testloader, 0):
        
        inputs, labels = data
        
        inputs = inputs.to(device)
        
        outs = model.forward(device)
        
        labels = np.eye(10)[labels]
        labels = torch.from_numpy(labels)
        labels = labels.to(device)
    
        running_acc += (outs == labels).float().sum()
    