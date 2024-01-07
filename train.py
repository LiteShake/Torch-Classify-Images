import torch
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def Train(model, trainset, epochs, device):
    
    Accuracy_lst = []
    Loss_lst = []
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 5e-4, betas=(.9, .999))
    loss_crit = CrossEntropyLoss()
    
    for epoch in range(epochs):
        
        running_loss = 0.0
        running_acc = 0.0
        
        for index, data in enumerate(trainset, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            
            
            optimizer.zero_grad()
            
            outs = model.forward(inputs)
            
            labels = np.eye(10)[labels]
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
            # print(labels)
            
            # print(outs.shape, labels.shape)
            
            loss = loss_crit(outs, labels)
            optimizer.step()
            
            running_acc += (outs == labels).float().sum()
            running_loss += loss.item()
            
            if index % 100 == 99:
                
                print(f"Epoch {epoch+1} \tAccuracy {running_acc:.3f} \tLoss {running_loss / 2000:.3f}")
                
                Accuracy_lst.append(running_acc)
                Loss_lst.append(running_loss)
                     
                running_loss = 0.0
                running_acc = 0.0
                
    return model, {"Accuracy":Accuracy_lst, "Loss":Loss_lst}
