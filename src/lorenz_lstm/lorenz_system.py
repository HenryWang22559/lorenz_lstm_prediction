"""
Lorenz system simulation module
"""
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, xyz, sigma=10.0, beta=8.0/3.0, rho=28.0):
    """
    Lorenz system differential equations
    """
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span, base_initial_xyz, num_trajectories, 
                                 perturbation_scale, num_points, 
                                 sigma=10.0, beta=8.0/3.0, rho=28.0):
    """
    Generates Lorenz system data with sensitivity to initial conditions.
    
    Args:
        t_span (tuple): Time span (t_start, t_end)
        base_initial_xyz (list): Base initial conditions [x0, y0, z0]
        num_trajectories (int): Number of trajectories to generate
        perturbation_scale (float): Scale of perturbation to initial conditions
        num_points (int): Number of points per trajectory
        sigma, beta, rho (float): Lorenz system parameters
    
    Returns:
        tuple: (trajectories array, time points array)
    """
    all_trajectories = []
    for _ in range(num_trajectories):
        initial_xyz = [
            base_initial_xyz[0] + np.random.uniform(-perturbation_scale, perturbation_scale),
            base_initial_xyz[1] + np.random.uniform(-perturbation_scale, perturbation_scale),
            base_initial_xyz[2] + np.random.uniform(-perturbation_scale, perturbation_scale)
        ]
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(lorenz, t_span, initial_xyz, 
                       args=(sigma, beta, rho), t_eval=t_eval)
        all_trajectories.append(sol.y.T)
    
    return np.array(all_trajectories), t_eval
