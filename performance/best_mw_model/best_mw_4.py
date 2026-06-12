import torch

device = torch.device('cuda')
dtype = torch.float32

from orbitflows import generate_sho_orbits
from orbitflows.dynamics import H_sho, H, MWPotential2014_1D
import torch

omega_guess = 1.1
omega_true = 1.5
t_end = 2 * torch.pi / omega_guess
n_steps = 256
r_bounds = torch.tensor([0.01, 0.75])
guess_ps, true_aa = generate_sho_orbits(n_orbits=16, omega=omega_guess, t_end=t_end, 
                    n_steps=n_steps, r_bounds=r_bounds)
guess_ps = guess_ps.to(device=device, dtype=dtype)
true_aa = true_aa.to(device=device, dtype=dtype)

from orbitflows import (HamiltonianMappingModel, 
                        SymplecticCouplingLayer, 
                        SimpleNNConditioner)
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define a learning rate scheduler
epsilon_float64 = torch.finfo(torch.float64).eps
scheduler = partial(ReduceLROnPlateau, mode='min', patience=50, 
                      factor=0.9, threshold=1e-15, threshold_mode='rel', 
                      eps=epsilon_float64)
# Define the model
targetPotential = MWPotential2014_1D
model = HamiltonianMappingModel(
    targetPotential=targetPotential,
    input_dim=2, 
    n_layers=256, 
    omega=omega_guess,
    layer_class=SymplecticCouplingLayer,
    conditioner=SimpleNNConditioner,
    conditioner_args={
        'num_layers' : 2,
        'projection_dims' : 64,
        'activation':torch.nn.ReLU,
        'include_bias':True},
    optimizer=torch.optim.Adam, 
    scheduler=scheduler
)
model.flow.to(device=device, dtype=dtype)

epochs = 100_000
model.train(guess_ps, epochs, lr=1e-3)

model.save('best_mw_3')

