import matplotlib.pyplot as plt
from solvers.basic_solver import *
from Helper.fluxes import *
from Models.CNN_model import *
torch.autograd.set_detect_anomaly(True)
# train_dataset_path="Data_generation/data/burgers_train_3s.pt"
train_dataset_path="Data_generation/data/burgers_train_1s.pt"
test_dataset_path="Data_generation/data/burgers_test_5s.pt"
dataset=torch.load(test_dataset_path)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

phi0=dataset["ic"]
phi_all=dataset["phi_all"]
dx=dataset["dx"]
dt=dataset["dt"]
Nx=dataset["Nx"]
Nt=dataset["Nt"]
t_max=dataset["t_max"]
x_min=dataset["x_min"]
x_max=dataset["x_max"]
model=CNN_model(device, 5, 5, [64, 64], activation=F.relu)
# model=CNN_model.load("Checkpoints/test4.pt")
# H_hat=H_hat_wrapperLF(H_Burgers)   
H_hat=model.forward 
total_epochs=100
epoch_group=20
best_loss=1e10
for i in range(total_epochs//epoch_group):
    best_loss=model.train_on_data(n_epochs=epoch_group, batch_size=32, lr=1e-4, dataset_path=train_dataset_path, save_folder="Checkpoints", save_name_prefix="test4", writer=None, best_loss=best_loss)
    solver=BasicSolver (H_hat, dx, dt, Nt, x_max, device)
    torch.cuda.empty_cache()
    U=solver.solve(phi0)
    rel_l2=torch.norm(U-phi_all)/torch.norm(phi_all)
    print(f"test Relative L2 error: {rel_l2:.4e}")



solver=BasicSolver( H_hat, dx, dt, Nt, x_max, device)
U=solver.solve(phi0)
rel_l2=torch.norm(U-phi_all)/torch.norm(phi_all)
print(f"Final Relative L2 error: {rel_l2:.4e}")
# Plot results
plt.figure(figsize=(12,5)) 

# Plot ground truth
plt.subplot(1,2,1)
plt.imshow(phi_all[0].cpu().detach().numpy(), aspect='auto', origin='lower', extent=[0, t_max, x_min, x_max])
plt.colorbar(label='Value')
plt.title('Ground Truth')
plt.xlabel('Time')
plt.ylabel('Space')

# Plot prediction
plt.subplot(1,2,2)
plt.imshow(U[0].cpu().detach().numpy(), aspect='auto', origin='lower', extent=[0, t_max, x_min, x_max])
plt.colorbar(label='Value')
plt.title('Model Prediction')
plt.xlabel('Time')
plt.ylabel('Space')

plt.tight_layout()
plt.savefig("Tests/test4.png")
plt.show()



