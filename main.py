import torch.nn as nn 
import train

from models import *
from datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainloader, testloader = LoadMNIST()
model = ConvSingleChan.ConvSingleChan()
print(model)

trained_model, history = train.Train(model, trainloader, 20, device)
