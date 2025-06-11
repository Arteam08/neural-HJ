import torch 

#define H(p) funcrtions

def H_Burgers(u):
    """
    Burgers Hamiltonian: H(u) = u^2/2
    """
    return u**2/2

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

def compute_divided_differences(u, order):
    """
    Compute divided differences up to given order.
    
    Args:
        u: input tensor (n_ic, Nx+1)
        order: order of divided differences to compute
    
    Returns:
        dd: list of divided differences tensors
    """
    dd = [u]
    for k in range(1, order + 1):
        prev = dd[-1]
        next_dd = (prev[:, 1:] - prev[:, :-1]) / k
        dd.append(next_dd)
    return dd

def H_hat_wrapperENO(H, order=3):
    """
    Returns the H_hat function implementing the ENO numerical flux
    
    Args:
        H: Hamiltonian function
        order: order of ENO reconstruction (default: 3)
    """
    def ENO_flux(U_x, t):
        # U_x: (n_ic, Nx+1, Nt)
        u_x = U_x[:, :, t-1]  # (n_ic, Nx+1)
        n_ic, Nx_plus_1 = u_x.shape
        Nx = Nx_plus_1 - 1
        
        # Compute divided differences
        dd = compute_divided_differences(u_x, order-1)
        
        # Initialize reconstructed values
        u_minus = torch.zeros((n_ic, Nx), device=u_x.device)
        u_plus = torch.zeros((n_ic, Nx), device=u_x.device)
        
        # ENO reconstruction
        for i in range(Nx):
            # Left reconstruction (u_{i+1/2}^-)
            left_stencil = i
            for k in range(1, order):
                if i + k < Nx and abs(dd[k][:, left_stencil]) > abs(dd[k][:, left_stencil - 1]):
                    left_stencil += 1
            
            # Right reconstruction (u_{i+1/2}^+)
            right_stencil = i + 1
            for k in range(1, order):
                if i + k < Nx and abs(dd[k][:, right_stencil]) > abs(dd[k][:, right_stencil - 1]):
                    right_stencil += 1
            
            # Compute reconstructed values using Newton interpolation
            u_minus[:, i] = u_x[:, i]
            u_plus[:, i] = u_x[:, i + 1]
            for k in range(1, order):
                u_minus[:, i] += dd[k][:, left_stencil - k:left_stencil].prod(dim=1)
                u_plus[:, i] += dd[k][:, right_stencil - k:right_stencil].prod(dim=1)
        
        # Compute numerical flux
        return (H(u_minus) + H(u_plus)) / 2
    
    return ENO_flux

def H_hat_wrapperWENO5(H):
    """
    Returns the H_hat function implementing the 5th order WENO numerical flux
    
    Args:
        H: Hamiltonian function
    """
    # WENO5 coefficients
    c_weights = torch.tensor([1/10, 6/10, 3/10], dtype=torch.float32)
    epsilon = 1e-6
    
    def linear_weights(v1, v2, v3, v4, v5):
        """Compute linear weights for WENO5"""
        s1 = 13/12 * (v1 - 2*v2 + v3)**2 + 1/4 * (v1 - 4*v2 + 3*v3)**2
        s2 = 13/12 * (v2 - 2*v3 + v4)**2 + 1/4 * (v2 - v4)**2
        s3 = 13/12 * (v3 - 2*v4 + v5)**2 + 1/4 * (3*v3 - 4*v4 + v5)**2
        
        alpha1 = c_weights[0] / (epsilon + s1)**2
        alpha2 = c_weights[1] / (epsilon + s2)**2
        alpha3 = c_weights[2] / (epsilon + s3)**2
        
        sum_alpha = alpha1 + alpha2 + alpha3
        
        return alpha1/sum_alpha, alpha2/sum_alpha, alpha3/sum_alpha
    
    def WENO_flux(U_x, t):
        # U_x: (n_ic, Nx+1, Nt)
        u_x = U_x[:, :, t-1]  # (n_ic, Nx+1)
        n_ic, Nx_plus_1 = u_x.shape
        Nx = Nx_plus_1 - 1
        
        # Pad for stencil
        u_padded = torch.nn.functional.pad(u_x, (2, 2), mode='replicate')
        
        # Initialize reconstructed values
        u_minus = torch.zeros((n_ic, Nx), device=u_x.device)
        u_plus = torch.zeros((n_ic, Nx), device=u_x.device)
        
        # WENO reconstruction
        for i in range(Nx):
            # Get stencil values
            v1, v2, v3, v4, v5 = [u_padded[:, i+j:i+j+1] for j in range(5)]
            
            # Compute nonlinear weights
            w1, w2, w3 = linear_weights(v1, v2, v3, v4, v5)
            
            # Compute candidate polynomials
            p1 = (2*v1 - 7*v2 + 11*v3) / 6
            p2 = (-v2 + 5*v3 + 2*v4) / 6
            p3 = (2*v3 + 5*v4 - v5) / 6
            
            # Compute reconstructed values
            u_minus[:, i] = w1*p1.squeeze() + w2*p2.squeeze() + w3*p3.squeeze()
            
            # Right-biased reconstruction (mirror of left-biased)
            v1, v2, v3, v4, v5 = [u_padded[:, i+j+1:i+j+2] for j in range(5)]
            w1, w2, w3 = linear_weights(v5, v4, v3, v2, v1)  # Note: reversed order
            
            p1 = (2*v5 - 7*v4 + 11*v3) / 6
            p2 = (-v4 + 5*v3 + 2*v2) / 6
            p3 = (2*v3 + 5*v2 - v1) / 6
            
            u_plus[:, i] = w1*p1.squeeze() + w2*p2.squeeze() + w3*p3.squeeze()
        
        # Compute numerical flux
        return (H(u_minus) + H(u_plus)) / 2
    
    return WENO_flux




