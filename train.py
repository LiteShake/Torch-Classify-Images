import torch
import numpy as np

def Train(model, optim, loss_crit, trainset, epochs, device):
    
    history = {
        'Accuracy':[],
        'Loss':[]
    }
    
    model = model.to(device)
    
    for epoch in range(epochs):
        
        running_loss = 0.0
        running_acc = 0.0
        
        for index, data in enumerate(trainset, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            
            optim.zero_grad()
            
            outs = model.forward(inputs)
            
            labels = np.eye(10)[labels]
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
            
            loss = loss_crit(outs, labels)
            optim.step()
            
            running_acc += (outs == labels).float().sum()
            running_loss += loss.item()
            
            if index % 100 == 99:
                
                print(f"Epoch {epoch+1} \tAccuracy {running_acc:.3f} \tLoss {running_loss / 2000:.3f}")
                
                history["Accuracy"] = history["Accuracy"].append(running_acc)
                history["Loss"] = history["Loss"].append(running_loss)
                     
                running_loss = 0.0
                running_acc = 0.0
                
    return model, history
