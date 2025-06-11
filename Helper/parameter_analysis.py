import torch
import matplotlib.pyplot as plt
from Data_generation.lh_burgers import generate_data_burgers
from solvers.basic_solver import BasicSolver
from Helper.fluxes import H_hat_wrapperLF, H_Burgers
from Helper.high_order_solvers import ENOSolver, WENOSolver
import os
import numpy as np
from Models.CNN_model import CNN_model

def get_default_base_params():
    """Returns default base parameters for the analysis."""
    return {
        'n_ic': 100,        # number of initial conditions
        'n_pieces': 3,      # number of pieces in piecewise function
       # number of time points
        't_max': 1.0,       # maximum time
        'x_min': 0.0,       # minimum x value
        'x_max': 1.0,       # maximum x value
        'dt': 5e-3,         # time step (derived from t_max/Nt but needed for some variations)
        'dx': 0.01,          # spatial step (derived from (x_max-x_min)/Nx but needed for some variations)
        'Nx': int((1.0-0.0)/0.01),
        'Nt': int(1.0/5e-3)
    }

def get_default_param_ranges():
    """Returns default parameter ranges for the analysis."""
    return {
        'dx': np.linspace(0.005, 0.02, 10),    # 10 different dx values
        'dt': np.linspace(0.001, 0.02, 10),    # 10 different dt values
        't_max': np.linspace(0.5, 10, 10)     # 10 different t_max values
    }

def analyze_parameter_sensitivity(model_checkpoint, base_params=None, param_ranges=None, save_folder="parameter_analysis"):
    """
    Analyze model sensitivity to different parameters (dx, dt, t_max) and compare with Lax-Friedrichs, ENO, and WENO.
    
    Args:
        model_checkpoint (str): Path to the model checkpoint
        base_params (dict, optional): Base parameters for data generation. If None, uses default values.
        param_ranges (dict, optional): Dictionary containing parameter ranges to test. If None, uses default ranges.
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
    dx_errors_lf = []
    dx_errors_eno = []
    dx_errors_weno = []
    
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
        
        # Neural Network solution
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.sqrt(torch.mean((U - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dx={dx} (Neural Network):")
            print(f"Relative L2 error: {rel_l2:.4e}")
            dx_errors.append(rel_l2.item())
            
        # Lax-Friedrichs solution
        solver_lf = BasicSolver(H_hat_wrapperLF(H_Burgers), dataset['dx'], dataset['dt'], 
                              dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_lf = solver_lf.solve(dataset['ic'])
            rel_l2_lf = torch.sqrt(torch.mean((U_lf - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dx={dx} (Lax-Friedrichs):")
            print(f"Relative L2 error: {rel_l2_lf:.4e}")
            dx_errors_lf.append(rel_l2_lf.item())
            
        # ENO solution
        solver_eno = ENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                             dataset['Nt'], dataset['x_max'], device, order=3)
        with torch.no_grad():
            U_eno = solver_eno.solve(dataset['ic'])
            rel_l2_eno = torch.sqrt(torch.mean((U_eno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dx={dx} (ENO):")
            print(f"Relative L2 error: {rel_l2_eno:.4e}")
            dx_errors_eno.append(rel_l2_eno.item())
            
        # WENO solution
        solver_weno = WENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                               dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_weno = solver_weno.solve(dataset['ic'])
            rel_l2_weno = torch.sqrt(torch.mean((U_weno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dx={dx} (WENO):")
            print(f"Relative L2 error: {rel_l2_weno:.4e}")
            dx_errors_weno.append(rel_l2_weno.item())
        
        os.remove(os.path.join(save_folder, f"temp_dx_{dx}.pt"))
    
    ax1.semilogy(param_ranges['dx'], dx_errors, 'o-', label='Neural Network')
    ax1.semilogy(param_ranges['dx'], dx_errors_lf, 'o--', label='Lax-Friedrichs')
    ax1.semilogy(param_ranges['dx'], dx_errors_eno, 's--', label='ENO')
    ax1.semilogy(param_ranges['dx'], dx_errors_weno, '^--', label='WENO')
    ax1.set_xlabel('dx')
    ax1.set_ylabel('Relative L2 Error (log scale)')
    ax1.set_title('Error vs dx')
    ax1.grid(True)
    ax1.legend()
    
    # Test dt variation
    dt_errors = []
    dt_errors_lf = []
    dt_errors_eno = []
    dt_errors_weno = []
    
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
        
        # Neural Network solution
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.sqrt(torch.mean((U - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dt={dt} (Neural Network):")
            print(f"Relative L2 error: {rel_l2:.4e}")
            dt_errors.append(rel_l2.item())
            
        # Lax-Friedrichs solution
        solver_lf = BasicSolver(H_hat_wrapperLF(H_Burgers), dataset['dx'], dataset['dt'], 
                              dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_lf = solver_lf.solve(dataset['ic'])
            rel_l2_lf = torch.sqrt(torch.mean((U_lf - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dt={dt} (Lax-Friedrichs):")
            print(f"Relative L2 error: {rel_l2_lf:.4e}")
            dt_errors_lf.append(rel_l2_lf.item())
            
        # ENO solution
        solver_eno = ENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                             dataset['Nt'], dataset['x_max'], device, order=3)
        with torch.no_grad():
            U_eno = solver_eno.solve(dataset['ic'])
            rel_l2_eno = torch.sqrt(torch.mean((U_eno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dt={dt} (ENO):")
            print(f"Relative L2 error: {rel_l2_eno:.4e}")
            dt_errors_eno.append(rel_l2_eno.item())
            
        # WENO solution
        solver_weno = WENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                               dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_weno = solver_weno.solve(dataset['ic'])
            rel_l2_weno = torch.sqrt(torch.mean((U_weno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for dt={dt} (WENO):")
            print(f"Relative L2 error: {rel_l2_weno:.4e}")
            dt_errors_weno.append(rel_l2_weno.item())
        
        os.remove(os.path.join(save_folder, f"temp_dt_{dt}.pt"))
    
    ax2.semilogy(param_ranges['dt'], dt_errors, 'o-', label='Neural Network')
    ax2.semilogy(param_ranges['dt'], dt_errors_lf, 'o--', label='Lax-Friedrichs')
    ax2.semilogy(param_ranges['dt'], dt_errors_eno, 's--', label='ENO')
    ax2.semilogy(param_ranges['dt'], dt_errors_weno, '^--', label='WENO')
    ax2.set_xlabel('dt')
    ax2.set_ylabel('Relative L2 Error (log scale)')
    ax2.set_title('Error vs dt')
    ax2.grid(True)
    ax2.legend()
    
    # Test t_max variation
    t_max_errors = []
    t_max_errors_lf = []
    t_max_errors_eno = []
    t_max_errors_weno = []
    
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
        
        # Neural Network solution
        solver = BasicSolver(model.forward, dataset['dx'], dataset['dt'], 
                           dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U = solver.solve(dataset['ic'])
            rel_l2 = torch.sqrt(torch.mean((U - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for t_max={t_max} (Neural Network):")
            print(f"Relative L2 error: {rel_l2:.4e}")
            t_max_errors.append(rel_l2.item())
            
        # Lax-Friedrichs solution
        solver_lf = BasicSolver(H_hat_wrapperLF(H_Burgers), dataset['dx'], dataset['dt'], 
                              dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_lf = solver_lf.solve(dataset['ic'])
            rel_l2_lf = torch.sqrt(torch.mean((U_lf - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for t_max={t_max} (Lax-Friedrichs):")
            print(f"Relative L2 error: {rel_l2_lf:.4e}")
            t_max_errors_lf.append(rel_l2_lf.item())
            
        # ENO solution
        solver_eno = ENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                             dataset['Nt'], dataset['x_max'], device, order=3)
        with torch.no_grad():
            U_eno = solver_eno.solve(dataset['ic'])
            rel_l2_eno = torch.sqrt(torch.mean((U_eno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for t_max={t_max} (ENO):")
            print(f"Relative L2 error: {rel_l2_eno:.4e}")
            t_max_errors_eno.append(rel_l2_eno.item())
            
        # WENO solution
        solver_weno = WENOSolver(H_Burgers, dataset['dx'], dataset['dt'], 
                               dataset['Nt'], dataset['x_max'], device)
        with torch.no_grad():
            U_weno = solver_weno.solve(dataset['ic'])
            rel_l2_weno = torch.sqrt(torch.mean((U_weno - dataset['phi_all'])**2)) / torch.sqrt(torch.mean(dataset['phi_all']**2))
            print(f"\nDiagnostics for t_max={t_max} (WENO):")
            print(f"Relative L2 error: {rel_l2_weno:.4e}")
            t_max_errors_weno.append(rel_l2_weno.item())
        
        os.remove(os.path.join(save_folder, f"temp_tmax_{t_max}.pt"))
    
    ax3.semilogy(param_ranges['t_max'], t_max_errors, 'o-', label='Neural Network')
    ax3.semilogy(param_ranges['t_max'], t_max_errors_lf, 'o--', label='Lax-Friedrichs')
    ax3.semilogy(param_ranges['t_max'], t_max_errors_eno, 's--', label='ENO')
    ax3.semilogy(param_ranges['t_max'], t_max_errors_weno, '^--', label='WENO')
    ax3.set_xlabel('t_max')
    ax3.set_ylabel('Relative L2 Error (log scale)')
    ax3.set_title('Error vs t_max')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'parameter_sensitivity.png'))
    plt.close()
    
    return {
        'dx_errors': dx_errors,
        'dx_errors_lf': dx_errors_lf,
        'dx_errors_eno': dx_errors_eno,
        'dx_errors_weno': dx_errors_weno,
        'dt_errors': dt_errors,
        'dt_errors_lf': dt_errors_lf,
        'dt_errors_eno': dt_errors_eno,
        'dt_errors_weno': dt_errors_weno,
        't_max_errors': t_max_errors,
        't_max_errors_lf': t_max_errors_lf,
        't_max_errors_eno': t_max_errors_eno,
        't_max_errors_weno': t_max_errors_weno
    }

analyze_parameter_sensitivity(model_checkpoint='Checkpoints/test4_prog3_best.pt')