import torch
import matplotlib.pyplot as plt
from Data_generation.lh_burgers import generate_data_burgers
from solvers.basic_solver import BasicSolver
import os
import numpy as np
from Models.CNN_model import CNN_model

def get_default_base_params():
    """Returns default base parameters for the analysis."""
    return {
        'n_ic': 100,        # number of initial conditions
        'n_pieces': 3,      # number of pieces in piecewise function
        'Nx': 100,          # number of spatial points
        'Nt': 100,          # number of time points
        't_max': 1.0,       # maximum time
        'x_min': 0.0,       # minimum x value
        'x_max': 1.0,       # maximum x value
        'dt': 0.01,         # time step (derived from t_max/Nt but needed for some variations)
        'dx': 0.01          # spatial step (derived from (x_max-x_min)/Nx but needed for some variations)
    }

def get_default_param_ranges():
    """Returns default parameter ranges for the analysis."""
    return {
        'dx': np.linspace(0.005, 0.02, 10),    # 10 different dx values
        'dt': np.linspace(0.005, 0.02, 10),    # 10 different dt values
        't_max': np.linspace(0.5, 2.0, 10)     # 10 different t_max values
    }

def analyze_parameter_sensitivity(model_checkpoint, base_params=None, param_ranges=None, save_folder="parameter_analysis"):
    """
    Analyze model sensitivity to different parameters (dx, dt, t_max).
    
    Args:
        model_checkpoint (str): Path to the model checkpoint
        base_params (dict, optional): Base parameters for data generation. If None, uses default values.
            - n_ic: number of initial conditions (default: 100)
            - n_pieces: number of pieces in piecewise function (default: 3)
            - Nx: number of spatial points (default: 100)
            - Nt: number of time points (default: 100)
            - t_max: maximum time (default: 1.0)
            - x_min: minimum x value (default: 0.0)
            - x_max: maximum x value (default: 1.0)
            - dt: time step (default: 0.01)
            - dx: spatial step (default: 0.01)
        param_ranges (dict, optional): Dictionary containing parameter ranges to test. If None, uses default ranges.
            - dx: list of dx values (default: np.linspace(0.005, 0.02, 10))
            - dt: list of dt values (default: np.linspace(0.005, 0.02, 10))
            - t_max: list of t_max values (default: np.linspace(0.5, 2.0, 10))
    """
    # Use default parameters if none provided
    if base_params is None:
        base_params = get_default_base_params()
    else:
        # Fill in any missing parameters with defaults
        default_params = get_default_base_params()
        for key in default_params:
            if key not in base_params:
                base_params[key] = default_params[key]

    if param_ranges is None:
        param_ranges = get_default_param_ranges()
    else:
        # Fill in any missing ranges with defaults
        default_ranges = get_default_param_ranges()
        for key in default_ranges:
            if key not in param_ranges:
                param_ranges[key] = default_ranges[key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_model.load(model_checkpoint)
    model.eval()
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Test dx variation
    dx_errors = []
    for dx in param_ranges['dx']:
        Nx = int((base_params['x_max'] - base_params['x_min']) / dx)
        test_data = generate_data_burgers(
            n_ic=base_params['n_ic'],
            n_pieces=base_params['n_pieces'],
            Nx=Nx,
            Nt=base_params['Nt'],
            t_max=base_params['t_max'],
            device=device,
            save_folder=save_folder,
            filename=f"temp_dx_{dx}.pt",
            x_min=base_params['x_min'],
            x_max=base_params['x_max']
        )
        
        dataset = torch.load(os.path.join(save_folder, f"temp_dx_{dx}.pt"), weights_only=False)
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.norm(U - dataset['phi_all']) / torch.norm(dataset['phi_all'])
            dx_errors.append(rel_l2.item())
        
        os.remove(os.path.join(save_folder, f"temp_dx_{dx}.pt"))
    
    ax1.plot(param_ranges['dx'], dx_errors, 'o-')
    ax1.set_xlabel('dx')
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_title('Error vs dx')
    ax1.grid(True)
    
    # Test dt variation
    dt_errors = []
    for dt in param_ranges['dt']:
        Nt = int(base_params['t_max'] / dt)
        test_data = generate_data_burgers(
            n_ic=base_params['n_ic'],
            n_pieces=base_params['n_pieces'],
            Nx=base_params['Nx'],
            Nt=Nt,
            t_max=base_params['t_max'],
            device=device,
            save_folder=save_folder,
            filename=f"temp_dt_{dt}.pt",
            x_min=base_params['x_min'],
            x_max=base_params['x_max']
        )
        
        dataset = torch.load(os.path.join(save_folder, f"temp_dt_{dt}.pt"), weights_only=False)
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.norm(U - dataset['phi_all']) / torch.norm(dataset['phi_all'])
            dt_errors.append(rel_l2.item())
        
        os.remove(os.path.join(save_folder, f"temp_dt_{dt}.pt"))
    
    ax2.plot(param_ranges['dt'], dt_errors, 'o-')
    ax2.set_xlabel('dt')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Error vs dt')
    ax2.grid(True)
    
    # Test t_max variation
    t_max_errors = []
    for t_max in param_ranges['t_max']:
        Nt = int(t_max / base_params['dt'])
        test_data = generate_data_burgers(
            n_ic=base_params['n_ic'],
            n_pieces=base_params['n_pieces'],
            Nx=base_params['Nx'],
            Nt=Nt,
            t_max=t_max,
            device=device,
            save_folder=save_folder,
            filename=f"temp_tmax_{t_max}.pt",
            x_min=base_params['x_min'],
            x_max=base_params['x_max']
        )
        
        dataset = torch.load(os.path.join(save_folder, f"temp_tmax_{t_max}.pt"), weights_only=False)
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.norm(U - dataset['phi_all']) / torch.norm(dataset['phi_all'])
            t_max_errors.append(rel_l2.item())
        
        os.remove(os.path.join(save_folder, f"temp_tmax_{t_max}.pt"))
    
    ax3.plot(param_ranges['t_max'], t_max_errors, 'o-')
    ax3.set_xlabel('t_max')
    ax3.set_ylabel('Relative L2 Error')
    ax3.set_title('Error vs t_max')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'parameter_sensitivity.png'))
    plt.close()
    
    return {
        'dx_errors': dx_errors,
        'dt_errors': dt_errors,
        't_max_errors': t_max_errors
    }

analyze_parameter_sensitivity(model_checkpoint='Checkpoints/test4_prog3_best.pt')