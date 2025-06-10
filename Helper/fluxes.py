import torch 

#define H(p) funcrtions

def H_Burgers(p):
    """Burgers equation H(p)=p*p/2"""
    return p*p/2

def H_prime_Burgers(p):
    """Derivative of Burgers Hamiltonian H'(p)=p"""
    return p

def H_hat_wrapperLF(H, alpha=None): 
    """
    Returns the H_hat function implementing the Lax-Friedrichs numerical flux
    
    Args:
        H: Hamiltonian function
        alpha: viscosity coefficient. If None, will be computed adaptively
    """
    def Lax_fried(U_x, t):
        # U_x: (n_ic, Nx+1, Nt)
        u_x = U_x[:, :, t-1]  # (n_ic, Nx+1)
        u_x_minus = u_x[:, :-1]  # (n_ic, Nx)
        u_x_plus = u_x[:, 1:]  # (n_ic, Nx)
        
        # Compute local alpha if not provided
        local_alpha = alpha if alpha is not None else torch.max(torch.abs(u_x))
        
        # Compute the numerical flux
        avg_term = H((u_x_minus + u_x_plus)/2)
        diff_term = local_alpha*(u_x_plus - u_x_minus)/2
        
        # Debug print every 1000 timesteps
        if t % 1000 == 0:
            print(f"\nNumerical Flux at t={t}:")
            print(f"Max u_x: {torch.max(abs(u_x)):.3e}")
            print(f"Max avg_term: {torch.max(abs(avg_term)):.3e}")
            print(f"Max diff_term: {torch.max(abs(diff_term)):.3e}")
            print(f"Alpha: {local_alpha}")
        
        return avg_term - diff_term
    
    return Lax_fried




