import torch
import matplotlib.pyplot as plt
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import matplotlib.animation as animation
######## THIS part is for picewise intitial conditions ########

# def generate_ceofficients(n_ic, n_pieces,device, x_min=0, x_max=1):
#     """generates a tensor (n_ic, n_pieces+1,3) with each time (x,a,b) the postiion of slope change, a the slope and b intercept, the last (a,b- are dummies)"""
#     # currently a nd b are just gaussians
#     x=torch.empty(n_ic, n_pieces+1,  device=device).uniform_(x_min, x_max) # (n_ic, n_pieces+1)
#     x=torch.sort(x, dim=1)[0] # (n_ic, n_pieces+1)
#     a=torch.randn(n_ic, n_pieces+1, device=device) # (n_ic, n_pieces+1)
#     b=torch.randn(n_ic, n_pieces+1, device=device) # (n_ic, n_pieces+1)
#     return torch.stack([x,a,b], dim=2) # (n_ic, n_pieces+1,3)

def generate_ceofficients(n_ic, n_pieces, device, x_min=0, x_max=1):
    """
    Generates a tensor (n_ic, n_pieces+1, 3) with (x, a, b) such that
    phi_0(x) is piecewise linear and continuous. The first x = x_min and
    the last x = x_max. Only n_pieces - 1 internal x values are random.
    """
    # Generate random internal breakpoints (n_pieces - 1), sort them
    x_internal = torch.empty(n_ic, n_pieces - 1, device=device).uniform_(x_min, x_max)
    x_internal = torch.sort(x_internal, dim=1)[0]

    # Concatenate fixed endpoints
    x_min_tensor = torch.full((n_ic, 1), x_min, device=device)
    x_max_tensor = torch.full((n_ic, 1), x_max, device=device)
    x = torch.cat([x_min_tensor, x_internal, x_max_tensor], dim=1)  # (n_ic, n_pieces + 1)

    # Generate random slopes
    a = torch.randn(n_ic, n_pieces + 1, device=device)

    # Initialize intercepts b
    b = torch.zeros(n_ic, n_pieces + 1, device=device)
    b[:, 0] = torch.randn(n_ic, device=device)  # random initial intercept

    # Enforce continuity: b_i = (a_{i-1} - a_i)*x_i + b_{i-1}
    for i in range(1, n_pieces + 1):
        b[:, i] = (a[:, i - 1] - a[:, i]) * x[:, i] + b[:, i - 1]

    return torch.stack([x, a, b], dim=2)  # (n_ic, n_pieces + 1, 3)




def phi0_piecewise(x, coefficients):
    """
    x: (Nx,) input points
    coefficients: (n_ic, n_pieces+1, 3): (x_pieces, a, b)
    Returns: (n_ic, Nx), where out-of-domain x values are assigned
    the boundary value (constant extrapolation).
    """
    n_ic, n_pieces_plus1, _ = coefficients.shape
    n_pieces = n_pieces_plus1 - 1
    Nx = x.shape[0]

    # Expand input x
    x = x.unsqueeze(0).expand(n_ic, Nx)  # (n_ic, Nx)

    # Extract piece data
    x_pieces = coefficients[:, :, 0]  # (n_ic, n_pieces+1)
    a = coefficients[:, :-1, 1]       # (n_ic, n_pieces)
    b = coefficients[:, :-1, 2]       # (n_ic, n_pieces)

    x_left = x_pieces[:, :-1]         # (n_ic, n_pieces)
    x_right = x_pieces[:, 1:]         # (n_ic, n_pieces)

    x_exp = x.unsqueeze(2)            # (n_ic, Nx, 1)
    mask = (x_exp >= x_left.unsqueeze(1)) & (x_exp < x_right.unsqueeze(1))  # (n_ic, Nx, n_pieces)

    # Handle x == x_max (right-closed domain)
    is_last = (x == x_pieces[:, -1].unsqueeze(1))  # (n_ic, Nx)
    mask[:, :, -1] |= is_last

    # Compute piecewise values
    a_exp = a.unsqueeze(1)  # (n_ic, 1, n_pieces)
    b_exp = b.unsqueeze(1)  # (n_ic, 1, n_pieces)
    phi_all = a_exp * x_exp + b_exp  # (n_ic, Nx, n_pieces)
    phi_piecewise = torch.sum(phi_all * mask, dim=2)  # (n_ic, Nx)

    # Compute boundary values at x_min and x_max
    x_min = x_pieces[:, 0]     # (n_ic,)
    x_max = x_pieces[:, -1]    # (n_ic,)
    phi_left = a[:, 0] * x_min + b[:, 0]            # (n_ic,)
    phi_right = a[:, -1] * x_max + b[:, -1]         # (n_ic,)

    # Broadcast for (n_ic, Nx)
    phi_left = phi_left.unsqueeze(1).expand(n_ic, Nx)
    phi_right = phi_right.unsqueeze(1).expand(n_ic, Nx)

    # Create out-of-domain masks
    left_mask = x < x_min.unsqueeze(1)    # (n_ic, Nx)
    right_mask = x > x_max.unsqueeze(1)   # (n_ic, Nx)

    # Fill in extrapolated values
    phi = phi_piecewise.clone()
    phi[left_mask] = phi_left[left_mask]
    phi[right_mask] = phi_right[right_mask]

    return phi




def phi_theorique_min_batch(x_grid, t, coefficients):
    """
    Computes phi(x, t) for all ICs using Hopf–Lax formula in batch.

    Parameters:
    -----------
    x_grid: (Nx,) tensor
    t: float (scalar)
    coefficients: (n_ic, n_pieces+1, 3) tensor of (x_pieces, a, b)

    Returns:
    --------
    phi_vals: (n_ic, Nx) tensor
    """
    n_ic, n_pieces_plus1, _ = coefficients.shape
    n_pieces = n_pieces_plus1 - 1
    Nx = x_grid.shape[0]

    # Get (n_ic, n_pieces)
    x_left = coefficients[:, :-1, 0]  # (n_ic, n_pieces)
    x_right = coefficients[:, 1:, 0]  # (n_ic, n_pieces)
    a = coefficients[:, :-1, 1]       # (n_ic, n_pieces)
    b = coefficients[:, :-1, 2]       # (n_ic, n_pieces)

    # Characteristic cone
    left = x_left + a * t   # (n_ic, n_pieces)
    right = x_right + a * t # (n_ic, n_pieces)

    # Broadcast x_grid
    x = x_grid.view(1, 1, Nx)              # (1, 1, Nx)
    left = left.unsqueeze(2)              # (n_ic, n_pieces, 1)
    right = right.unsqueeze(2)
    x_left = x_left.unsqueeze(2)
    x_right = x_right.unsqueeze(2)
    a = a.unsqueeze(2)
    b = b.unsqueeze(2)

    # Compute piecewise cases (n_ic, n_pieces, Nx)
    inside = (x >= left) & (x <= right)
    left_of = x < left
    right_of = x > right

    phi_inside = -0.5 * a**2 * t + a * x + b
    phi_left = x_left * a + b + ((x - x_left) ** 2) / (2 * t)
    phi_right = x_right * a + b + ((x_right - x) ** 2) / (2 * t)

    phi_all = (
        inside * phi_inside +
        left_of * phi_left +
        right_of * phi_right
    )  # (n_ic, n_pieces, Nx)

    # Take min over intervals
    phi_min, _ = torch.min(phi_all, dim=1)  # (n_ic, Nx)
    return phi_min







def generate_data_burgers(n_ic, n_pieces, Nx, Nt, t_max, device,save_folder, filename, x_min=0.0, x_max=1.0):
    """
    Generate (phi0, phi_all) via Hopf–Lax formula for multiple ICs in parallel.
    """
    x_grid = torch.linspace(x_min, x_max, Nx, device=device)
    t_grid = torch.linspace(0, t_max, Nt, device=device)

    coefs = generate_ceofficients(n_ic, n_pieces, device, x_min, x_max)  # (n_ic, n_pieces+1, 3)

    # Initial condition at t=0
    phi0 = phi0_piecewise(x_grid, coefs)  # (n_ic, Nx)

    # Solution for all t
    phi_all = torch.empty(n_ic, Nx, Nt, device=device)
    for k, t in enumerate(t_grid):
        if t == 0:
            phi_all[:, :, k] = phi0
        else:
            phi_all[:, :, k] = phi_theorique_min_batch(x_grid, t.item(), coefs)

    phi0, phi_all  # Optionally save to disk
    dx=(x_max-x_min)/Nx
    dt=t_max/Nt
    dataset={"ic":phi0, "phi_all":phi_all, "dx":dx, "dt":dt, "Nx":Nx, "Nt":Nt, "n_pieces":n_pieces, "n_ic":n_ic, "x_min":x_min, "x_max":x_max, "t_max":t_max}
    save_path=os.path.join(save_folder, filename)
    torch.save(dataset, save_path)
    return phi0, phi_all












# # === Generate data ===
# phi0, phi_all = generate_data_burgers(
#     n_ic=1, n_pieces=4, Nx=Nx, Nt=Nt, t_max=t_max, device=device, save_folder=save_folder, filename=filename
# )  # phi0: (1, Nx), phi_all: (1, Nx, Nt)

# plt.plot(phi0[0].cpu().numpy())

# plt.savefig("phi0.png")

# plt.plot(phi_all[0,:,10].cpu().numpy())
# plt.savefig("phi_all.png")

# phi_all = phi_all[0].cpu()  # shape: (Nx, Nt)
# x_grid = torch.linspace(0, 1, phi_all.shape[0]).cpu()
# Nt = phi_all.shape[1]

# # === Set up animation ===
# fig, ax = plt.subplots()
# line, = ax.plot(x_grid, phi_all[:, 0])
# ax.set_ylim(phi_all.min().item(), phi_all.max().item())
# ax.set_title("Time evolution of φ(x, t)")
# ax.set_xlabel("x")
# ax.set_ylabel("φ(x, t)")

# def update(frame):
#     line.set_ydata(phi_all[:, frame])
#     ax.set_title(f"t = {frame / (Nt - 1) * t_max:.3f} s")
#     return line,

# ani = animation.FuncAnimation(
#     fig, update, frames=Nt, blit=True, interval=200  # 50 ms per frame ~ 20 FPS
# )

# # === Save video ===
# ani.save("phi_evolution.mp4", writer='ffmpeg', fps=20)
# plt.close(fig)

# print("✅ Saved phi_evolution.mp4")


