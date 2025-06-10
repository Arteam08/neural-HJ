import torch
import matplotlib.pyplot as plt


#we define a class that solves the equation give H, H^, dx, dt, t_steps
torch.autograd.set_detect_anomaly(True)

clipping_value=0.5e3

class BasicSolver:
    def __init__(self,  H_hat, dx, dt, Nt, x_max, device):

        self.H_hat = H_hat # H_hat(U_x, t): (n_ic, Nx+1, Nt), 1 -> (n_ic, Nx)
        self.dx = dx
        self.dt = dt
        self.x_max = x_max #  we solve on [0, x_max]
        self.Nx=int(x_max/dx)
        self.Nt = Nt
        self.device = device

    def get_U_x_t(self, u):
        """gets the derivative for 1 timestep u: (n_ic, Nx) -> (n_ic, Nx+1, Nt)"""
        n_ic=u.shape[0]
        u_x=torch.zeros(n_ic, self.Nx+1, device=self.device)
        u_x[:,1:-1]=(u[:,1:]-u[:,:-1])/self.dx
        u_x[:,0]=u_x[:,1] # ghost cells
        u_x[:,-1]=u_x[:,-2] # ghost cells
        return u_x

    def solve(self, ic):
        """
        Solves the HJ equation using Euler method
        ic : (n_ic, Nx) - initial condition
        Returns: U (n_ic, Nx, Nt) - solution at all time steps
        """
        n_ic=ic.shape[0]
        U=torch.zeros((n_ic, self.Nx,   self.Nt), device=self.device) # adapt the image to a square
        U_x=torch.zeros((n_ic, self.Nx+1, self.Nt), device=self.device)
        U[:,:,0]=ic
        U_x[:,:,0]=self.get_U_x_t(U[:,:,0])
        for t in range(1, self.Nt):
            H_temp=self.H_hat(U_x, t)
            H_temp=torch.clamp(H_temp, min=-clipping_value, max=clipping_value)
            assert torch.all(torch.isfinite(H_temp)), "H_temp is not finite"
            U[:,:,t]=U[:,:,t-1] -self.dt*H_temp
            U=torch.clamp(U, min=-clipping_value, max=clipping_value)
            assert torch.all(torch.isfinite(U[:,:,t])), "U is not finite"
            U_x[:,:,t]=self.get_U_x_t(U[:,:,t])
            U_x=torch.clamp(U_x, min=-clipping_value, max=clipping_value)
            assert torch.all(torch.isfinite(U_x[:,:,t])), "U_x is not finite"
        return U

    
    
        