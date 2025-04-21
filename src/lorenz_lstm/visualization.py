"""
Visualization utilities for Lorenz system predictions
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_3d_trajectories(actual_trajectories, predicted_trajectories, initial_sequences=None, num_plots=3):
    """
    Plot 3D comparison of actual vs predicted trajectories
    
    Args:
        actual_trajectories (list): List of actual trajectory arrays
        predicted_trajectories (list): List of predicted trajectory arrays
        initial_sequences (list): List of initial sequence arrays
        num_plots (int): Number of trajectories to plot
    """
    fig = plt.figure(figsize=(15, 5 * ((num_plots + 2) // 3)))
    
    for i in range(min(num_plots, len(actual_trajectories))):
        ax = fig.add_subplot(((num_plots + 2) // 3), 3, i + 1, projection='3d')
        
        # Plot actual trajectory
        actual = actual_trajectories[i]
        ax.scatter(actual[:, 0], actual[:, 1], actual[:, 2], 
                  marker='o', color='blue', label='Actual', s=10)
        
        # Plot predicted trajectory
        predicted = predicted_trajectories[i]
        ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2],
                color='red', label='Predicted')
        
        # Plot initial sequence if provided
        if initial_sequences is not None:
            initial = initial_sequences[i]
            ax.plot(initial[:, 0], initial[:, 1], initial[:, 2],
                   color='green', label='Initial')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Trajectory {i+1}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_time_series(t, actual_trajectory, predicted_trajectory, sequence_length):
    """
    Plot time series comparison of actual vs predicted values
    
    Args:
        t (array): Time points
        actual_trajectory (array): Actual trajectory values
        predicted_trajectory (array): Predicted trajectory values
        sequence_length (int): Length of input sequence
    """
    fig = plt.figure(figsize=(12, 8))
    components = ['X', 'Y', 'Z']
    
    for i in range(3):
        ax = fig.add_subplot(3, 1, i + 1)
        
        # Plot actual values
        ax.plot(t[:len(actual_trajectory)], actual_trajectory[:, i],
                label=f'Actual {components[i]}')
        
        # Plot predicted values
        t_pred = t[sequence_length:sequence_length + len(predicted_trajectory)]
        ax.plot(t_pred, predicted_trajectory[:, i], '--',
                label=f'Predicted {components[i]}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(components[i])
        ax.legend()
    
    plt.tight_layout()
    plt.show()
