import torch
import torchvision
import torchvision.transforms as transforms

def LoadCIFAR10() :
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # datasets
    trainset = torchvision.datasets.CIFAR10(
        './data',
        download=True,
        train=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        './data',
        download=True,
        train=False,
        transform=transform
    )

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=0)


    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                            shuffle=False, num_workers=0)
    
    return trainloader, testloader