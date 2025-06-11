import torch
from solvers.basic_solver import BasicSolver

class ENOSolver(BasicSolver):
    """
    ENO solver for Hamilton-Jacobi equations.
    Uses ENO reconstruction for computing accurate spatial derivatives.
    """
    def __init__(self, H, dx, dt, Nt, x_max, device, order=3):
        self.H = H  # Store Hamiltonian function
        super().__init__(lambda u, t: u, dx, dt, Nt, x_max, device)  # Dummy forward function
        self.order = order

    def compute_eno_coefficients(self, v, order):
        """
        Compute ENO coefficients for derivative approximation.
        v: stencil values
        order: order of accuracy
        """
        n_ic = v.shape[0]
        coeffs = torch.zeros((n_ic, order), device=self.device)
        
        # First order differences (first derivative approximations)
        D1 = (v[:, 1:] - v[:, :-1]) / self.dx
        
        if order == 1:
            return D1[:, 0:1]
            
        # Second order differences
        D2 = (D1[:, 1:] - D1[:, :-1]) / self.dx
        
        if order == 2:
            # Choose the least oscillatory stencil
            k = (torch.abs(D2[:, 0]) < torch.abs(D2[:, 1])).long()
            coeffs[:, 0] = D1[torch.arange(n_ic), k]
            coeffs[:, 1] = D2[torch.arange(n_ic), k] / 2
            return coeffs
            
        # Third order differences
        D3 = (D2[:, 1:] - D2[:, :-1]) / self.dx
        
        if order == 3:
            # Choose the least oscillatory stencil
            k0 = (torch.abs(D2[:, 0]) < torch.abs(D2[:, 1])).long()
            k = k0.clone()
            # Update k based on D3 comparison
            mask = torch.abs(D3[torch.arange(n_ic), k]) >= torch.abs(D3[torch.arange(n_ic), k+1])
            k[mask] += 1
            
            coeffs[:, 0] = D1[torch.arange(n_ic), k]
            coeffs[:, 1] = D2[torch.arange(n_ic), k] / 2
            coeffs[:, 2] = D3[torch.arange(n_ic), k] / 6
            return coeffs
            
        return coeffs

    def get_derivatives(self, phi):
        """
        Compute left and right derivatives using ENO reconstruction
        """
        n_ic = phi.shape[0]
        
        # Pad for stencil calculations
        pad_width = self.order + 1
        phi_padded = torch.nn.functional.pad(phi, (pad_width, pad_width), mode='replicate')
        
        # Initialize derivatives
        p_minus = torch.zeros((n_ic, self.Nx), device=self.device)
        p_plus = torch.zeros((n_ic, self.Nx), device=self.device)
        
        for i in range(self.Nx):
            # Extract stencil for left derivative (p⁻)
            stencil_minus = phi_padded[:, i:i+2*pad_width-1]
            coeffs_minus = self.compute_eno_coefficients(stencil_minus, self.order)
            p_minus[:, i] = coeffs_minus.sum(dim=1)
            
            # Extract stencil for right derivative (p⁺)
            stencil_plus = phi_padded[:, i+1:i+2*pad_width]
            coeffs_plus = self.compute_eno_coefficients(stencil_plus, self.order)
            p_plus[:, i] = coeffs_plus.sum(dim=1)
        
        return p_minus, p_plus

    def solve(self, phi0):
        """
        Solves the HJ equation using ENO reconstruction for derivatives
        phi0: (n_ic, Nx) - initial condition
        Returns: phi (n_ic, Nx, Nt) - solution at all time steps
        """
        n_ic = phi0.shape[0]
        phi = torch.zeros((n_ic, self.Nx, self.Nt), device=self.device)
        phi[:, :, 0] = phi0
        
        for t in range(1, self.Nt):
            # Compute derivatives using ENO reconstruction
            p_minus, p_plus = self.get_derivatives(phi[:, :, t-1])
            
            # Compute Hamiltonian
            H_minus = self.H(p_minus)
            H_plus = self.H(p_plus)
            
            # Choose upwind direction
            H_num = torch.where(
                p_minus * p_plus > 0,
                torch.where(p_minus > 0, H_minus, H_plus),
                torch.minimum(H_minus, H_plus)
            )
            
            # Add CFL condition for stability
            max_H = torch.max(torch.abs(H_num))
            dt_cfl = self.dt * torch.minimum(torch.tensor(1.0), 1.0 / (max_H + 1e-6))
            
            phi[:, :, t] = phi[:, :, t-1] - dt_cfl * H_num
            
        return phi

class WENOSolver(BasicSolver):
    """
    WENO solver for Hamilton-Jacobi equations.
    Uses WENO reconstruction for computing accurate spatial derivatives.
    """
    def __init__(self, H, dx, dt, Nt, x_max, device):
        self.H = H  # Store Hamiltonian function
        super().__init__(lambda u, t: u, dx, dt, Nt, x_max, device)  # Dummy forward function
        
        # WENO parameters
        self.epsilon = 1e-6  # Smoothness parameter
        
        # Linear weights (optimal weights for smooth solutions)
        self.c = torch.tensor([0.1, 0.6, 0.3], device=device)
        
        # Reconstruction coefficients for 3 stencils
        self.a_minus = torch.tensor([
            [1/3, -7/6, 11/6],      # S0
            [-1/6, 5/6, 1/3],       # S1
            [1/3, 5/6, -1/6]        # S2
        ], device=device)
        
        self.a_plus = torch.tensor([
            [-1/6, 5/6, 1/3],       # S0
            [1/3, 5/6, -1/6],       # S1
            [11/6, -7/6, 1/3]       # S2
        ], device=device)

    def compute_smoothness_indicators(self, v):
        """
        Compute smoothness indicators β for WENO weights
        v: stencil values [batch, 5]
        """
        # Scale differences by dx for proper scaling
        dx = self.dx
        dx2 = dx * dx
        
        v0, v1, v2, v3, v4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
        
        # First term: measure of first derivative variation
        β0 = 13/12 * ((v0 - 2*v1 + v2)/dx2)**2 + 1/4 * ((v0 - 4*v1 + 3*v2)/dx)**2
        β1 = 13/12 * ((v1 - 2*v2 + v3)/dx2)**2 + 1/4 * ((v1 - v3)/dx)**2
        β2 = 13/12 * ((v2 - 2*v3 + v4)/dx2)**2 + 1/4 * ((3*v2 - 4*v3 + v4)/dx)**2
        
        return torch.stack([β0, β1, β2], dim=1)

    def compute_nonlinear_weights(self, β):
        """
        Compute nonlinear weights ω from smoothness indicators
        """
        # Add small constant to β for numerical stability
        τ5 = torch.abs(β[:, 0] - β[:, 2])  # 5th order smoothness indicator
        β = β + self.epsilon + τ5.unsqueeze(1)
        
        # Compute α weights
        α = self.c / (β**2)
        
        # Normalize weights
        ω = α / α.sum(dim=1, keepdim=True)
        return ω

    def get_derivatives(self, phi):
        """
        Compute left and right derivatives using WENO reconstruction
        """
        n_ic = phi.shape[0]
        
        # Pad for stencil calculations
        phi_padded = torch.nn.functional.pad(phi, (2, 2), mode='replicate')
        
        # Initialize derivatives
        p_minus = torch.zeros((n_ic, self.Nx), device=self.device)
        p_plus = torch.zeros((n_ic, self.Nx), device=self.device)
        
        for i in range(self.Nx):
            # Extract 5-point stencil
            v = phi_padded[:, i:i+5]
            
            # Compute smoothness indicators
            β = self.compute_smoothness_indicators(v)
            
            # Compute nonlinear weights
            ω_minus = self.compute_nonlinear_weights(β)
            ω_plus = self.compute_nonlinear_weights(β.flip(dims=[1]))
            
            # Compute candidate approximations for p⁻
            p0_minus = (self.a_minus[0] * v[:, :3]).sum(dim=1) / self.dx
            p1_minus = (self.a_minus[1] * v[:, 1:4]).sum(dim=1) / self.dx
            p2_minus = (self.a_minus[2] * v[:, 2:5]).sum(dim=1) / self.dx
            
            # Compute candidate approximations for p⁺
            p0_plus = (self.a_plus[0] * v[:, :3]).sum(dim=1) / self.dx
            p1_plus = (self.a_plus[1] * v[:, 1:4]).sum(dim=1) / self.dx
            p2_plus = (self.a_plus[2] * v[:, 2:5]).sum(dim=1) / self.dx
            
            # Combine using nonlinear weights
            p_minus[:, i] = (ω_minus[:, 0] * p0_minus + 
                           ω_minus[:, 1] * p1_minus + 
                           ω_minus[:, 2] * p2_minus)
            
            p_plus[:, i] = (ω_plus[:, 0] * p0_plus + 
                          ω_plus[:, 1] * p1_plus + 
                          ω_plus[:, 2] * p2_plus)
        
        return p_minus, p_plus

    def solve(self, phi0):
        """
        Solves the HJ equation using WENO reconstruction for derivatives
        phi0: (n_ic, Nx) - initial condition
        Returns: phi (n_ic, Nx, Nt) - solution at all time steps
        """
        n_ic = phi0.shape[0]
        phi = torch.zeros((n_ic, self.Nx, self.Nt), device=self.device)
        phi[:, :, 0] = phi0
        
        for t in range(1, self.Nt):
            # Compute derivatives using WENO reconstruction
            p_minus, p_plus = self.get_derivatives(phi[:, :, t-1])
            
            # Compute Hamiltonian
            H_minus = self.H(p_minus)
            H_plus = self.H(p_plus)
            
            # Choose upwind direction
            H_num = torch.where(
                p_minus * p_plus > 0,
                torch.where(p_minus > 0, H_minus, H_plus),
                torch.minimum(H_minus, H_plus)
            )
            
            # Add CFL condition for stability
            max_H = torch.max(torch.abs(H_num))
            dt_cfl = self.dt * torch.minimum(torch.tensor(1.0), 1.0 / (max_H + 1e-6))
            
            phi[:, :, t] = phi[:, :, t-1] - dt_cfl * H_num
            
        return phi 