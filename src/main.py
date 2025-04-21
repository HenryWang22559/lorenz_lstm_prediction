"""
Main script for training and testing the Lorenz LSTM model
"""
import numpy as np
import torch

from lorenz_lstm.lorenz_system import generate_lorenz_data
from lorenz_lstm.data import prepare_data_loaders
from lorenz_lstm.model import LorenzLSTM
from lorenz_lstm.train import train_model
from lorenz_lstm.visualization import (
    plot_training_history,
    plot_3d_trajectories,
    plot_time_series
)

def main():
    # Parameters
    t_span = (0, 30)
    base_initial_xyz = [1, 1, 1]
    num_trajectories = 200
    perturbation_scale = 10
    num_points = 3000
    sequence_length = 10
    batch_size = 500
    num_epochs = 70
    
    # Generate data
    print("Generating Lorenz system data...")
    lorenz_data, t_eval = generate_lorenz_data(
        t_span, base_initial_xyz, num_trajectories,
        perturbation_scale, num_points
    )
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        lorenz_data, sequence_length, batch_size
    )
    
    # Initialize and train model
    print("Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LorenzLSTM().to(device)
    history = train_model(model, train_loader, val_loader, num_epochs, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    # Generate predictions for visualization
    print("Generating predictions...")
    num_test_trajectories = 3
    num_predictions = 200
    
    actual_trajectories = []
    predicted_trajectories = []
    initial_sequences = []
    
    # Get some test sequences
    test_data = iter(test_loader)
    for i in range(num_test_trajectories):
        x, y = next(test_data)
        initial_sequence = x[0].unsqueeze(0)  # Add batch dimension
        
        # Generate predictions
        predictions = model.predict_sequence(initial_sequence, num_predictions)
        predicted_trajectory = np.array(predictions)
        
        # Get corresponding actual trajectory
        actual_trajectory = lorenz_data[-(i+1)]  # Use last few trajectories
        actual_trajectory = actual_trajectory[:num_predictions + sequence_length]
        
        actual_trajectories.append(actual_trajectory)
        predicted_trajectories.append(predicted_trajectory)
        initial_sequences.append(initial_sequence[0].numpy())
    
    # Plot results
    plot_3d_trajectories(actual_trajectories, predicted_trajectories, initial_sequences)
    
    # Plot time series for the first trajectory
    plot_time_series(t_eval, actual_trajectories[0], predicted_trajectories[0], sequence_length)

if __name__ == "__main__":
    main()
