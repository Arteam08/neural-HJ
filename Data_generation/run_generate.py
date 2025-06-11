import torch
from lh_burgers import generate_data_burgers
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_min=0.0
x_max=1.0
n_ic=1000
n_pieces=4 # nombre de discontinuities
dx=1e-2
dt=5*1e-3
t_max=0.1
Nx=int((x_max-x_min)/dx)
Nt=int(t_max/dt)


device=device
save_folder="Data_generation/data"
filename="burgers_train_0p1s.pt"
generate_data_burgers(n_ic, n_pieces, Nx, Nt, t_max, device, save_folder, filename, x_min, x_max)