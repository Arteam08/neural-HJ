import matplotlib.pyplot as plt
from solvers.basic_solver import *
from Helper.fluxes import *
from Models.CNN_model import *
# Set up dataloader for batched evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
test_dataset_path="Data_generation/data/burgers_train_1s.pt"
dataset = torch.load(test_dataset_path)
burgers_dataset=BurgersDataset(dataset, device)
data_loader = DataLoader(burgers_dataset, batch_size=batch_size, shuffle=False)


phi0=dataset["ic"]
phi_all=dataset["phi_all"]
dx=dataset["dx"]
dt=dataset["dt"]
Nx=dataset["Nx"]
Nt=dataset["Nt"]
t_max=dataset["t_max"]
x_min=dataset["x_min"]
x_max=dataset["x_max"]


model=CNN_model.load("Checkpoints/test4.pt")
H_hat=model.forward 
solver=BasicSolver (H_hat, dx, dt, Nt, x_max, device)

total_rel_l2 = 0
num_batches = 0

with torch.no_grad():
    for ic, U_GT in data_loader:
        U = solver.solve(ic)
        rel_l2 = torch.norm(U - U_GT) / torch.norm(U_GT)
        total_rel_l2 += rel_l2
        num_batches += 1

avg_rel_l2 = total_rel_l2 / num_batches
print(f"test Relative L2 error: {avg_rel_l2:.4e}")








