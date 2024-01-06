import torch
import torchvision
import torchvision.transforms as transforms

def LoadMNIST() :
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # datasets
    trainset = torchvision.datasets.MNIST(
        './data',
        download=True,
        train=True,
        transform=transform
    )
    testset = torchvision.datasets.MNIST(
        './data',
        download=True,
        train=False,
        transform=transform
    )

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)


    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader