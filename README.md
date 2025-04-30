# Lorenz System LSTM Predictor

This project uses LSTM (Long Short-Term Memory) to predict the behavior of the Lorenz system, a classic example of chaos theory. The model learns to predict future states of the system given a sequence of past states, demonstrating how deep learning can capture chaotic dynamics.

## About the Lorenz System

The Lorenz system is a system of ordinary differential equations known for demonstrating chaotic behavior:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

where σ = 10, ρ = 28, and β = 8/3 are the standard parameters that produce chaos.

## Model Overview

- **Architecture**: LSTM network with 32 hidden units
- **Input**: Sequences of 10 time steps, each containing (x, y, z) coordinates
- **Output**: Prediction of the next (x, y, z) state
- **Training Data**: Generated using numerical integration (RK45 method)

## Project Structure

```
.
├── src/
│   ├── lorenz_lstm/
│   │   ├── lorenz_system.py    # Lorenz system simulation
│   │   ├── data.py             # Data preparation
│   │   ├── model.py            # LSTM model
│   │   ├── train.py            # Training logic
│   │   └── visualization.py     # Plotting tools
│   └── main.py                 # Main script
└── requirements.txt            # Dependencies
```

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model:
```bash
python src/main.py
```

The script will:
- Generate Lorenz system trajectories
- Train the LSTM model
- Show predictions vs actual trajectories

## Results

The model demonstrates:
- Accurate short-term predictions
- Capture of the butterfly-shaped attractor
- Understanding of the system's sensitivity to initial conditions

Visualization includes:
- 3D trajectory comparisons
- Time series plots of x, y, z coordinates
- Training loss curves
