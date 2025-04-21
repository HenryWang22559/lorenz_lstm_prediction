"""
Data preparation module for Lorenz system prediction
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class LorenzDataset(Dataset):
    """Dataset class for Lorenz system data"""
    def __init__(self, trajectories, sequence_length):
        self.data = []
        self.targets = []
        self._prepare_data(trajectories, sequence_length)
        
    def _prepare_data(self, trajectories, sequence_length):
        """Prepare sequential data from trajectories"""
        for traj in trajectories:
            for i in range(len(traj) - sequence_length):
                self.data.append(traj[i:i + sequence_length])
                self.targets.append(traj[i + sequence_length])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])

def prepare_sequence_data_loaders(trajectories, sequence_length, batch_size=32, train_split=0.8, val_split=0.1):
    """
    Prepare DataLoaders for training, validation, and testing
    
    Args:
        trajectories (np.ndarray): Array of trajectories
        sequence_length (int): Length of input sequences
        batch_size (int): Batch size for training
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset = LorenzDataset(trajectories, sequence_length)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
