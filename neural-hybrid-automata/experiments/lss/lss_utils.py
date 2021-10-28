import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class NominalVectorField(nn.Module):
    """Nominal vector field with embedded switching modes. Can be integrated with any standard ODE solver. 
        The stiffness of the problem can make adaptive-stepping slow around switching surfaces.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, t, z):
        x, y = z
        dx, dy = torch.zeros_like(x), torch.zeros_like(y)
        
        # Switching conditions
        idx1 = x >= 2
        idx2 = (x < 2) * (y >= 0)
        idx3 = (x < 2) * (y < 0)
        
        # Compute vector field
        dx[idx1], dy[idx1] = -y[idx1], x[idx1] + 2
        dx[idx2], dy[idx2] = -torch.ones_like(x[idx2]), -torch.ones_like(y[idx2])
        dx[idx3], dy[idx3] = torch.ones_like(x[idx3]), -torch.ones_like(y[idx3])
        
        return (dx, dy)

        
def sample_at_swichting_surf(n_sample=128, x_lim=[-1, 3], y_lim=[-15, 15], Δ=0.1):
    "Sample initial conditions around the switching surfaces."
    x0 = torch.Tensor(n_sample//2, 1).uniform_(x_lim, 2-Δ)
    y0 = torch.Tensor(n_sample//2, 1).uniform_(-Δ, Δ)
    x1 = torch.Tensor(n_sample//2, 1).uniform_(2-Δ, 2+Δ)
    y1 = torch.Tensor(n_sample//2, 1).uniform_(*y_lim)
    return torch.cat([x0, x1]), torch.cat([y0, y1])

def label_trajectories(x, y):
    """Label a trajectory (x_t, y_t) according to its mode."""
    label = torch.zeros(x.shape, device=x.device, dtype=torch.int64)
    idx1 = x >= 2
    idx2 = (x < 2) * (y >= 0)
    idx3 = (x < 2) * (y < 0)
    label[idx1] = 0
    label[idx2] = 1
    label[idx3] = 2
    return label

def loss_fn(sol_x, sol_y, xT, yT):
    "Compute L_inf loss on trajectories"
    e_x = torch.abs(sol_x - xT)
    e_y = torch.abs(sol_y - yT)
    loss = (e_x + e_y).mean()
    
    # L_2 loss on finite differences (approx. vector field)
    vf_x = (sol_x[1:] - sol_x[:-1] - xT[1:] + xT[:-1])**2
    vf_y = (sol_y[1:] - sol_y[:-1] - yT[1:] + yT[:-1])**2
    loss = loss + 0.5*(vf_x + vf_y).mean()
    return loss

def plot_modes(g, itr, device):
    n_grid = 100
    x, y = torch.linspace(-1.5, 3.5, n_grid), torch.linspace(-3.5, 3.5, n_grid)
    X, Y = torch.meshgrid(x, y); z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], -1).to(device)
    W = nn.Softmax(-1)(g(z)).detach().cpu()
    w_1, w_2, w_3, w_4, w_5 = W[:, :1], W[:, 1:2], W[:, 2:3], W[:, 3:4], W[:, 4:5]

    fig, axs = plt.subplots(1, 4, figsize=(21, 4))
    axs[0].contourf(X, Y, w_1.reshape(n_grid, n_grid), 100, cmap='inferno')
    axs[1].contourf(X, Y, w_2.reshape(n_grid, n_grid), 100, cmap='inferno')
    axs[2].contourf(X, Y, w_3.reshape(n_grid, n_grid), 100, cmap='inferno')
    axs[3].contourf(X, Y, w_4.reshape(n_grid, n_grid), 100, cmap='inferno')

    axs[0].plot([2, 2], [-3.5, 3.5], '--', c='blue')
    axs[0].set_xlim([-1.5, 3.5]) ; axs[0].set_ylim([-3.5, 3.5])
    axs[0].set_xlabel('x') ; axs[0].set_ylabel('y') ; axs[0].set_title('w_1')

    axs[1].plot([2, 2], [0, 3.5], '--', c='blue')
    axs[1].plot([-1.5, 2], [0, 0], '--b')
    axs[1].set_xlim([-1.5, 3.5]) ; axs[1].set_ylim([-3.5, 3.5])
    axs[1].set_xlabel('x') ; axs[1].set_ylabel('y') ; axs[1].set_title('w_2')

    axs[2].plot([2, 2], [-3.5, 0], '--', c='blue')
    axs[2].plot([-1.5, 2], [0, 0], '--b')
    axs[2].set_xlim([-1.5, 3.5]) ; axs[2].set_ylim([-3.5, 3.5])
    axs[2].set_xlabel('x') ; axs[2].set_ylabel('y') ; axs[2].set_title('w_3')
    plt.savefig(f'boundary_ste_{itr}.jpg')
    plt.close()