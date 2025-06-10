import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from solvers.basic_solver import BasicSolver
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)

class BurgersDataset(Dataset):
    def __init__(self, dataset_dict, device):
        self.ic = dataset_dict["ic"].to(device)
        self.phi_all = dataset_dict["phi_all"].to(device)
        
    def __len__(self):
        return len(self.ic)
    
    def __getitem__(self, idx):
        return self.ic[idx], self.phi_all[idx]


class CNN_model(nn.Module):
    def __init__(self, device, x_stencil, t_stencil, hidden_dims, activation=F.relu, **kwargs):
        super().__init__()  # Call parent's init with no arguments
        self.device = device
        self.x_stencil = x_stencil  # spatial stencil size
        self.t_stencil = t_stencil  # temporal stencil size
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.epoch_count=0

        in_channels = self.t_stencil # temporal dimension becomes channels

        layers = [
            nn.Conv1d(in_channels, hidden_dims[0], kernel_size=self.x_stencil, bias=True),  # (batch_size, hidden_dims[0], Nx+1-x_stencil+1)
            nn.ReLU()
        ]
        for i in range(len(hidden_dims) - 1):
            layers += [
                nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=1, bias=True),      # (batch_size, hidden_dims[i+1], Nx+1-x_stencil+1)
                nn.ReLU()
            ]
        layers.append(nn.Conv1d(hidden_dims[-1], 1, kernel_size=1, bias=True))    # (batch_size, 1, Nx+1-x_stencil+1)

        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, U_x, t, **kwargs): 
        """ computes the flux for the t+1 step using the t index"""
        batch_size, Nxp1, Nt = U_x.shape     # (batch_size, Nx+1, Nt)

        # Create new tensor for history
        if t >= self.t_stencil-1:
            U_hist = torch.clone(U_x[:, :, t - self.t_stencil+1 : t+1])
        else:
            pad = torch.clone(U_x[:, :, 0:1]).expand(-1, -1, self.t_stencil - t)
            U_hist = torch.cat([pad, torch.clone(U_x[:, :, :t])], dim=2)

        # Pad spatial dimension
        pad_right = (self.x_stencil-2) // 2
        pad_left = self.x_stencil - pad_right - 2
        U_hist = F.pad(U_hist, pad=(0, 0, pad_left, pad_right), mode='replicate')
        
        # Ensure contiguous memory and proper shape
        U_hist = U_hist.permute(0, 2, 1).contiguous()
        out = self.net(U_hist)
        out = out.squeeze(1)
        
        return out

    def save(self, checkpoint_folder, save_name):
        os.makedirs(checkpoint_folder, exist_ok=True)
        path = os.path.join(checkpoint_folder, save_name + ".pt")

        save_data = {
            "model_type": "CNN",
            "state_dict": self.state_dict(),
            "init_params": {
                "x_stencil": self.x_stencil,
                "t_stencil": self.t_stencil,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation.__name__,
                "device": self.device,
            }
        }
        torch.save(save_data, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path,map_location=torch.device("cpu"))

        params = checkpoint["init_params"]

        # Map string name to actual activation function
        activation_map = {
            "relu": F.relu,
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
            # Extend as needed
        }
        activation_fn = activation_map.get(params["activation"])
        if activation_fn is None:
            raise ValueError(f"Unknown activation function: {params['activation']}")

        model = cls(
            device=params["device"],
            x_stencil=params["x_stencil"],
            t_stencil=params["t_stencil"],
            hidden_dims=params["hidden_dims"],
            activation=activation_fn, 
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(params["device"])
        return model

    def train_on_data(self, 
                      n_epochs,
                      batch_size,
                      lr,
                      dataset_path, 
                      save_folder, 
                      save_name_prefix,
                      scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                      best_loss=float("inf"),
                      writer=None,
                      loss_fn=nn.MSELoss(),
                      ):
        self.train()
        dataset_dict = torch.load(dataset_path)
        dataset = BurgersDataset(dataset_dict, self.device)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)

        scheduler = scheduler_class(optimizer, mode='min', patience=10, factor=0.5)

        if writer is None:
            log_dir = os.path.join("runs", save_name_prefix)
            writer = SummaryWriter(log_dir=log_dir)

        solver = BasicSolver(self.forward, dataset_dict["dx"], dataset_dict["dt"], 
                           dataset_dict["Nt"], dataset_dict["x_max"], self.device)

        for e in range(n_epochs):
            self.epoch_count += 1
            epoch_loss = 0
            epoch_rel_l2 = 0

            for i, (ic, U_GT) in enumerate(data_loader):
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Ensure tensors require grad
                ic = ic.requires_grad_(True)
                U = solver.solve(ic)
                loss = loss_fn(U, U_GT)
                rel_l2 = loss.item() / torch.norm(U_GT)

                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()

                epoch_loss += loss.item()
                epoch_rel_l2 += rel_l2

            epoch_loss /= len(data_loader)
            epoch_rel_l2 /= len(data_loader)

            scheduler.step(epoch_loss)

            if e % 5 == 0:
                print(f"Epoch {self.epoch_count}/{n_epochs} | Loss: {epoch_loss:.4e} | RelL2: {epoch_rel_l2:.4e}")
                self.save(save_folder, save_name_prefix)
                writer.add_scalar("Loss/train", epoch_loss, self.epoch_count)
                writer.add_scalar("RelL2/train", epoch_rel_l2, self.epoch_count)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], self.epoch_count)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(save_folder, save_name_prefix + "_best")
                print(f"Best model saved at epoch {self.epoch_count} with loss {epoch_loss:.4e}")

        return best_loss
