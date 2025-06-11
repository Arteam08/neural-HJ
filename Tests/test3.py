import matplotlib.pyplot as plt
from solvers.basic_solver import *
from Helper.fluxes import *


dataset_path="Data_generation/data/burgers_data.pt"
dataset=torch.load(dataset_path)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

phi0=dataset["ic"]
phi_all=dataset["phi_all"]
dx=dataset["dx"]
dt=dataset["dt"]
Nx=dataset["Nx"]
Nt=dataset["Nt"]
t_max=dataset["t_max"]
x_min=dataset["x_min"]
x_max=dataset["x_max"]

H_hat=H_hat_wrapperLF(H_Burgers)    

solver=BasicSolver(H_Burgers, H_hat, dx, dt, Nt, x_max)
phi_tensor_bis=solver.solve(phi0)


# plot the solutions next to each other
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(phi_all[0,:,:].cpu().detach().numpy(), aspect='auto', extent=[0, t_max, x_min, x_max])
plt.colorbar(label='u(x,t)')
plt.title('Reference Solution')
plt.xlabel('x')
plt.ylabel('t')

plt.subplot(1, 2, 2)
plt.imshow(phi_tensor_bis[0,:,:].cpu().detach().numpy(), aspect='auto', extent=[0, t_max, x_min, x_max])
plt.colorbar(label='u(x,t)')
plt.title('Numerical Solution')
plt.xlabel('t')
plt.ylabel('x')

plt.suptitle('Hamilton-Jacobi Equation Solutions', fontsize=12)
plt.tight_layout()
plt.savefig("Tests/test3.png")
plt.show()


# Move tensors to the same device and compute difference
phi_tensor_bis = phi_tensor_bis.to(device)
phi_all = phi_all.to(device)
diff = phi_tensor_bis - phi_all
rel_l2 = torch.sqrt(torch.mean((phi_tensor_bis - phi_all)**2)) / torch.sqrt(torch.mean(phi_all**2))
print(f"Relative L2 error: {rel_l2:.4e}")










