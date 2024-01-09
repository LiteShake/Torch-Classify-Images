import torch.nn as nn 
from Trainer import Trainer

from models import *
from datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trainloader, testloader = LoadMNIST()
# model = ConvSingleChan.ConvSingleChan()

trainloader, testloader = LoadCIFAR10()
model = ConvThreeChan.ConvThreeChan()

print(model)

modelTrainer = Trainer(model.cuda())
trained_model, history = modelTrainer.Train(trainloader, 20, device)
