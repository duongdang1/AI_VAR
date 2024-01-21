import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                       test_dir:str,
                       transform: transforms.Compose,
                       batch_size:int,
                       num_workers: int= NUM_WORKERS):
    """Create dataloaders for training and testing sets from the given directories."""
    
    
    # create PyTorch training and validation datasets
    # without writing customs classes
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data  = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    # provide an iterable over a dataset, 
    # making it easy access batches of data for training
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size, #divides the data into batches, allowing processing over smaller samples at a time
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader,test_dataloader,class_names