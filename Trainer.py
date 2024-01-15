import torch
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    def __init__(self, model):
        
        self.model = model
        
    
    def Train(self, trainset, epochs, device, model_name):
        
        writer = SummaryWriter(f'runs/{model_name}')
        
        Accuracy_lst = []
        Loss_lst = []
        
        self.model.cuda()
        print(f"Using device {next(self.model.parameters()).device}")
        
        optimizer = optim.Adam(self.model.parameters(), lr = 5e-4, betas=(.9, .999))
        loss_crit = CrossEntropyLoss()
        
        for epoch in range(epochs):
            
            running_loss = 0.0
            running_acc = 0.0
            
            for index, data in enumerate(trainset, 0):
                
                for batch in data :
                    
                    inputs, labels = data
                    inputs = inputs.to(device)
                    
                    
                    optimizer.zero_grad()
                    
                    outs = self.model.forward(inputs)
                    
                    labels = np.eye(10)[labels]
                    labels = np.array([labels])
                    labels = torch.from_numpy(labels)
                    labels = labels.to(device)
                    # print(labels)
                    
                    # print(outs, labels)
                    
                    loss = loss_crit(outs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_acc += (outs == labels).float().sum()
                    running_loss += loss.item()
                    
                    
                if index % 1000 == 999:    
                    
                    print(f"Epoch {epoch+1} \tAccuracy {running_acc:.3f} \tLoss {running_loss / 2000:.3f}")
                    
                    Accuracy_lst.append(running_acc)
                    Loss_lst.append(running_loss)
                        
                    running_loss = 0.0
                    running_acc = 0.0
                
        return self.model, {"Accuracy":Accuracy_lst, "Loss":Loss_lst}
