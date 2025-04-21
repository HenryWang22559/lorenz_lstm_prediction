"""
LSTM model for Lorenz system prediction
"""
import torch
import torch.nn as nn

class LorenzLSTM(nn.Module):
    """LSTM model for predicting Lorenz system trajectories"""
    def __init__(self, input_size=3, hidden_size=32, output_size=3):
        """
        Initialize the LSTM model
        
        Args:
            input_size (int): Number of input features (3 for x, y, z)
            hidden_size (int): Number of LSTM hidden units
            output_size (int): Number of output features (3 for x, y, z)
        """
        super(LorenzLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])  # Take only the last time step
        return out
    
    def predict_sequence(self, initial_sequence, num_steps):
        """
        Predict future trajectory points
        
        Args:
            initial_sequence (torch.Tensor): Initial sequence of shape (1, sequence_length, input_size)
            num_steps (int): Number of future steps to predict
            
        Returns:
            list: Predicted trajectory points
        """
        self.eval()
        with torch.no_grad():
            current_sequence = initial_sequence
            predictions = []
            
            for _ in range(num_steps):
                next_point = self(current_sequence)
                predictions.append(next_point[0].numpy())
                
                # Update sequence for next prediction
                current_sequence = torch.cat([
                    current_sequence[:, 1:],
                    next_point.unsqueeze(1)
                ], dim=1)
                
        return predictions
