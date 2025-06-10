import torch
import matplotlib.pyplot as plt
from Helper.fluxes import *
from solvers.basic_solver import *

# Parameters for the simulation
dx = 0.1
dt = 0.01
Nt = 500  # Reduced for faster testing
x_max = 2*torch.pi  # Using 2π domain for periodic IC
Nx = int(x_max/dx)
n_ic = 1

# Create numerical flux with adaptive alpha
H_hat = H_hat_wrapperLF(H_Burgers)  # Let alpha be computed adaptively

solver = BasicSolver(H_Burgers, H_hat, dx, dt, Nt, x_max)

# Create initial condition
x_values = torch.linspace(0.0, x_max, Nx)
ic = -torch.cos(x_values*4)  # Using -cos(x) as IC
ic = ic.unsqueeze(0)

print("Initial condition range:", torch.min(ic).item(), torch.max(ic).item())
print("Initial derivative range:", torch.min(ic[:,1:] - ic[:,:-1]).item()/dx, 
      torch.max(ic[:,1:] - ic[:,:-1]).item()/dx)

# Solve
U = solver.solve(ic)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Space-time evolution
im = ax1.imshow(U[0,:,:].T, aspect='auto', 
                extent=[0, x_max, 0, dt*Nt],
                origin='lower')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_title('Space-time evolution')
plt.colorbar(im, ax=ax1, label='u(x,t)')

# Plot 2: Solution at different times
times_to_plot = [0, int(Nt/4), int(Nt/2), int(3*Nt/4), Nt-1]
for t in times_to_plot:
    ax2.plot(x_values, U[0,:,t], label=f't={t*dt:.2f}')
ax2.set_xlabel('x')
ax2.set_ylabel('u(x,t)')
ax2.set_title('Solution profiles at different times')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('Tests/test1.png', bbox_inches='tight', dpi=300)
plt.show()

# Print diagnostic information
print(f"\nFinal diagnostics:")
print(f"Solution range: [{torch.min(U):.3f}, {torch.max(U):.3f}]")
print(f"Total simulation time: {dt*Nt:.3f}")
print(f"dx/dt ratio: {dx/dt:.3f}")
print(f"Max change between any timesteps: {torch.max(abs(U[:,:,1:] - U[:,:,:-1])):.3e}")






