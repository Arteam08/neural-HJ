from Data_generation.lh_burgers import *
import matplotlib.pyplot as plt
from solvers.basic_solver import *
from Helper.fluxes import *

n_ic=1

T=0.5
x_range=(0.0, 1.0)
dx=0.1
dt=0.02
x_max=1.0

H_hat=H_hat_wrapperLF(H_Burgers)

phi_tensor, phi0_tensor, x_vals, t_vals = generate_laxhopf_tensor_torch(n_ic, Nx, Nt, T, x_range=x_range)

# solve it uusng the basic solver
solver=BasicSolver(H_Burgers, H_hat, dx, dt, Nt, x_max)
phi_tensor_bis=solver.solve(phi0_tensor)

#plot the solutions next to each other
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(phi_tensor[0,:,:], aspect='auto', extent=[0, 1, 0, 1])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(phi_tensor_bis[0,:,:], aspect='auto', extent=[0, 1, 0, 1])
plt.colorbar()
plt.savefig('Tests/test2.png')
plt.show()