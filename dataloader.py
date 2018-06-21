import os
import torch
from torchvision import transforms, datasets


def load_datasets(input_size, args):
    """
    load the dataloader according to args and input size.
    """
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # precomputed stats
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(224), # disable for now
        #transforms.RandomHorizontalFlip(), #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(180),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=8,
                                             pin_memory=True)
    return train_loader, val_loader