from orbitflows import HamiltonianMappingModel, generate_sho_orbits
from orbitflows.integrate.correction import dH_dx
from orbitflows.integrate import eulerstep, hamiltonian_fixed_angle
from orbitflows.integrate import rungekutta4 as rk4
import matplotlib.pyplot as plt
import numpy as np
import torch
from orbitflows import H
from time import time
from functools import partial 
from tqdm import tqdm


h_fixed_angle = partial(hamiltonian_fixed_angle, theta_set=torch.tensor(0.0))

def isoDiskPotential(x, amp=1, sigma=0.1):
    _H = sigma / torch.sqrt(torch.tensor([8.0 * torch.pi * amp]))[0]
    _sigma2 = sigma**2
    return 2.0 * _sigma2 * torch.log(torch.cosh(0.5 * x / _H))

# functions to generate plots

def step_size_cmap(model, stepsizes):
    '''
    Create a color map for the parameter space of integration and correction step sizes
    '''

    def _get_grid_data():
        rk4_std_matrix = np.zeros((len(stepsizes), len(stepsizes)))
        euler_std_matrix = np.zeros((len(stepsizes), len(stepsizes)))
        rk4_error_matrix = np.zeros((len(stepsizes), len(stepsizes)))
        euler_error_matrix = np.zeros((len(stepsizes), len(stepsizes)))
        for i, dt in enumerate(stepsizes):
            for j, ss in enumerate(stepsizes):
                steps = int(t_end / dt)
                aa_rk4_ = model.integrate(aa_, steps, t_end, correction_step_size=ss, correction=rk4, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
                aa_euler_ = model.integrate(aa_, steps, t_end, correction_step_size=ss, hamiltonian_tilde=h_fixed_angle, correction=eulerstep).to(torch.float64)
                euler_percent_error_ = np.mean(100 * (np.abs(model.hamiltonian(aa_euler_).detach() - H0_.detach()) /  H0_.detach()).numpy())
                euler_std_ = np.std(model.hamiltonian(aa_euler_).detach().numpy())
                rk4_percent_error_ = np.mean(100 * (np.abs(model.hamiltonian(aa_rk4_).detach() - H0_.detach()) /H0_.detach()).numpy())
                rk4_std_= np.std(model.hamiltonian(aa_rk4_).detach().numpy())
                rk4_std_matrix[i, j] = rk4_std_
                rk4_error_matrix[i, j] = rk4_percent_error_
                euler_std_matrix[i, j] = euler_std_
                euler_error_matrix[i, j] = euler_percent_error_
        return rk4_std_matrix, euler_std_matrix, rk4_error_matrix, euler_error_matrix

    rk4_std_matrix, euler_std_matrix, rk4_error_matrix, euler_error_matrix = _get_grid_data()

    # initialize figure and axes
    fig, axes = plt.subplots(2, 5, figsize=(18, 10), sharex='col', gridspec_kw={'width_ratios': [1, 1, 0.1, 1, 0.1]})

    # Define common extent for all plots
    extent = [np.log10(stepsizes.min()), np.log10(stepsizes.max()), 
            np.log10(stepsizes.min()), np.log10(stepsizes.max())]
    interp_tech = 'none'
    # Standard deviation plots (top row)
    vmin_std = min(np.min(rk4_std_matrix), np.min(euler_std_matrix))
    vmax_std = max(np.max(rk4_std_matrix), np.max(euler_std_matrix))

    im1 = axes[0, 0].imshow(rk4_std_matrix, extent=extent, origin='lower', 
                            aspect='auto', cmap='inferno', vmin=vmin_std, vmax=vmax_std, interpolation=interp_tech)
    axes[0, 0].set_title('RK4 $\sigma_E$', fontsize=20)
    axes[0, 0].set_ylabel('log10(dt)', fontsize=15)

    im2 = axes[0, 1].imshow(euler_std_matrix, extent=extent, origin='lower', 
                            aspect='auto', cmap='inferno', vmin=vmin_std, vmax=vmax_std, interpolation=interp_tech)
    axes[0, 1].set_title('Euler $\sigma_E$', fontsize=20)

    # Calculate ratio for std deviation (RK4/Euler)
    std_ratio = rk4_std_matrix/euler_std_matrix
    im5 = axes[0, 3].imshow(std_ratio, extent=extent, origin='lower',
                        aspect='auto', cmap='RdBu', vmin=0.5, vmax=1.5, interpolation=interp_tech)
    axes[0, 3].set_title('RK4/Euler $\sigma_E$ Ratio', fontsize=20)

    # Error percentage plots (bottom row)
    vmin_err = min(np.min(rk4_error_matrix), np.min(euler_error_matrix))
    vmax_err = max(np.max(rk4_error_matrix), np.max(euler_error_matrix))

    im3 = axes[1, 0].imshow(rk4_error_matrix, extent=extent, origin='lower', 
                            aspect='auto', cmap='inferno', vmin=vmin_err, vmax=vmax_err, interpolation=interp_tech)
    axes[1, 0].set_title('RK4 $\%$ Error', fontsize=20)
    axes[1, 0].set_xlabel('log(Correction Step Size)', fontsize=15)
    axes[1, 0].set_ylabel('log(dt)', fontsize=15)

    im4 = axes[1, 1].imshow(euler_error_matrix, extent=extent, origin='lower', 
                            aspect='auto', cmap='inferno', vmin=vmin_err, vmax=vmax_err, interpolation=interp_tech)
    axes[1, 1].set_title('Euler $\%$ Error ', fontsize=20)
    axes[1, 1].set_xlabel('log(Correction Step Size)', fontsize=15)

    # Calculate ratio for error percentage (RK4/Euler)
    error_ratio = rk4_error_matrix/euler_error_matrix
    im6 = axes[1, 3].imshow(error_ratio, extent=extent, origin='lower',
                        aspect='auto', cmap='RdBu', vmin=0.5, vmax=1.5, interpolation=interp_tech)
    axes[1, 3].set_title('RK4/Euler $\%$  Error Ratio', fontsize=20)
    axes[1, 3].set_xlabel('log(Correction Step Size)', fontsize=15)


    # Add space between columns
    fig.subplots_adjust(wspace=0.2, right=0.85)
    for ax in [axes[0,2], axes[1,2]]:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])

    for ax in [axes[0,3], axes[1,3]]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])

    for ax in [axes[0,4], axes[1,4]]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.03, pos.y0, pos.width, pos.height])

    # Create colorbars between columns 2 and 3
    # Colorbar for top row (std deviation)
    #cax1 = fig.add_axes([0.46, 0.55, 0.02, 0.3])
    cbar1 = fig.colorbar(im1, cax=axes[0,2])
    cbar1.set_label('$\sigma_E$', fontsize=15)

    # Colorbar for bottom row (error percentage)
    #cax2 = fig.add_axes([0.46, 0.15, 0.02, 0.3])
    cbar2 = fig.colorbar(im3, cax=axes[1,2])
    cbar2.set_label('Error (%)', fontsize=15)

    # Add colorbars on the right side for ratio plots
    #cax3 = fig.add_axes([0.87, 0.55, 0.02, 0.3])
    cbar3 = fig.colorbar(im5, cax=axes[0,4])
    cbar3.set_label('$\sigma_{E, RK4}$ / $\sigma_{E, euler}$', fontsize=15)

    #cax4 = fig.add_axes([0.87, 0.15, 0.02, 0.3])
    cbar4 = fig.colorbar(im6, cax=axes[1,4])
    cbar4.set_label('$\%$ Error RK4 / $\%$ Error Euler', fontsize=15)

    return fig, ax


def example_orbits_and_derivatives(model, training_data, n_grid_points=100):
    '''
    Create a plot of the orbits and derivatives of the model     
    '''
    def freq_tilde(model, aa):
        '''
        Compute the frequency of the system using the hamiltonian_tilde function.
        
        Parameters
        ----------
        aa : torch.tensor
            action-angle variables

        Returns
        -------
        torch.tensor
            frequency of the system assuming the hamiltonian_tilde function
        '''
        theta, j = aa[..., 0], aa[..., 1]
        _aa = torch.stack((theta, j), dim=-1)
        return torch.autograd.grad(h_fixed_angle(model, _aa), j, allow_unused=True)[0]
    def h_error(model, ps):
        '''
        Compute the error of the model prediction in the Hamiltonian.
        
        Parameters
        ----------
        ps : torch.tensor
            phase-space point
        
        Returns
        -------
        torch.tensor
            Hamiltonian error of the model prediction
        '''
        _aa = model.ps_to_aa(ps)
        return H(ps, model.targetPotential) - h_fixed_angle(model, _aa)
    
    def run_grid(model, ps_grid, n_grid_points):
        dH_dq_grid = []
        dH_dp_grid = []
        for ps_ in tqdm(ps_grid.requires_grad_()):
            dH_dq_grid.append(dH_dx(ps_, 'q', partial(h_error, model)))
            dH_dp_grid.append(dH_dx(ps_, 'p', partial(h_error, model)))
        # Reshape error_values to the grid shape
        dH_dq_grid = torch.tensor(dH_dq_grid).reshape(n_grid_points, n_grid_points)
        dH_dp_grid = torch.tensor(dH_dp_grid).reshape(n_grid_points, n_grid_points)
        return dH_dq_grid, dH_dp_grid
    
    def example_orbit(model, aa0):
        return model.integrate(aa0, 20, 10, correction=rk4, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
    ex_dt = 10/20
    ps0 = torch.tensor([1.0, 0], requires_grad=True).to(torch.float64)
    H0 = H(ps0, model.targetPotential)
    aa0 = model.ps_to_aa(ps0)
    aa_rk4 = example_orbit(model, aa0)

    scaling = 1.25
    x_min, x_max = training_data[..., 0].min().item() * scaling, training_data[..., 0].max().item() * scaling
    vx_min, vx_max = training_data[..., 1].min().item() * scaling, training_data[..., 1].max().item() * scaling

    # Number of points for each dimension in the grid
    x_minmax = torch.tensor(torch.abs(torch.tensor([x_min, x_max]))).max().item()
    vx_minmax = torch.tensor(torch.abs(torch.tensor([vx_min, vx_max]))).max().item()
    x_grid = torch.linspace(-x_minmax, x_minmax, n_grid_points)
    vx_grid = torch.linspace(-vx_minmax, vx_minmax, n_grid_points)
    X, VX = torch.meshgrid(x_grid, vx_grid, indexing='xy')
    ps_grid = torch.stack([X.flatten(), VX.flatten()], dim=1).to(torch.float64)
    dH_dq_grid, dH_dp_grid = run_grid(model, ps_grid, n_grid_points)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.075})
    pcm = ax[0].pcolormesh(X.numpy(), VX.numpy(), dH_dq_grid.numpy(), cmap='RdBu', shading='auto')
    pcm2 = ax[1].pcolormesh(X.numpy(), VX.numpy(), dH_dp_grid.numpy(), cmap='RdBu', shading='auto')

    ax[0].set_title('$- \\frac{\partial{H_{error}}(q,p)}{\partial q}$', fontsize=20)
    ax[1].set_title('$\\frac{\partial{H_{error}}(q,p)}{\partial p}$', fontsize=20)

    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    plt.colorbar(pcm, cax=cax)
    minmax = torch.abs(torch.tensor([dH_dq_grid.max().item(), dH_dq_grid.min().item(), dH_dp_grid.max().item(), dH_dp_grid.min().item()])).max().item()
    pcm.set_clim(vmin=-minmax, vmax=minmax)
    pcm2.set_clim(vmin=-minmax, vmax=minmax)

    freq_tildes = torch.tensor([freq_tilde(model, aa_) for aa_ in aa_rk4]).to(torch.float64) # Shape (n_steps, 2)
    aa_rk4_guess = aa_rk4.clone().detach() # Shape (n_steps, 2)
    aa_rk4_guess[...,0] = aa_rk4[..., 0] +  torch.tensor(freq_tildes) * ex_dt
    nf_prediction_ps = model.aa_to_ps(aa_rk4_guess.to(torch.float64)) # Shape (n_steps, 2)
    q_plot_quiver = model.aa_to_ps(aa_rk4)[..., 0]
    p_plot_quiver = model.aa_to_ps(aa_rk4)[..., 1]
    delta_q_guess = nf_prediction_ps[..., 0] - q_plot_quiver
    delta_p_guess = nf_prediction_ps[..., 1] - p_plot_quiver

    ex_orb = aa0.repeat(len(aa_guess[0][...,0]), 1)
    ex_orb[...,0] = aa_guess[0][...,0].detach()

    for axi in ax:
        axi.scatter(model.aa_to_ps(aa_guess)[..., 0].detach().numpy(),
            model.aa_to_ps(aa_guess)[..., 1].detach().numpy(),
            c='k', alpha=0.1, s=1)
        axi.plot(*model.aa_to_ps(aa_rk4).T.detach().numpy(), 'b', alpha=0.5, label='example orbit, integrated with flow (rk4)', lw=1)
        axi.scatter(*model.aa_to_ps(aa_rk4).T.detach().numpy(), c='b', alpha=0.75, s=20)
        axi.quiver(q_plot_quiver[:-1].detach().numpy(), p_plot_quiver[:-1].detach().numpy(),
                delta_q_guess.detach().numpy()[:-1], delta_p_guess.detach().numpy()[:-1],
                angles='xy', scale_units='xy', scale=1, color='k', width=0.005, alpha=0.65,
                label=f'Uncorrected Step')
        
    ax[1].annotate('q', xy=(-0.04, -0.15), xycoords='axes fraction', ha='center', fontsize=20)
    ax[0].annotate('p', xy=(-0.15, 0.5), xycoords='axes fraction', ha='center', fontsize=20, rotation='vertical')
    ax[0].set_xlim(-1.5, 1.5)
    ax[0].set_ylim(-1.5, 1.5)
    ax[1].legend(fontsize=10, loc='upper right')
    ax[1].annotate('Well Trained Model, smaller correction steps', xy=(-0.04, 1.2), xycoords='axes fraction', ha='center', fontsize=20)
    return fig, ax


def error_vs_loss_plot(rk4_error_big_list, euler_error_big_list, rk4_error_small_list, euler_error_small_list, loss_list, dt_big, dt_small, stride):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(loss_list[::stride], rk4_error_big_list, c='r', label=f'RK4, dt={dt_big:.2f}')
    ax.plot(loss_list[::stride], euler_error_big_list, c='b', label=f'Euler, dt={dt_big:.2f}')
    ax.plot(loss_list[::stride], rk4_error_small_list, c='r', linestyle='--', label=f'RK4, dt={dt_small:.2f}')
    ax.plot(loss_list[::stride], euler_error_small_list, c='b', linestyle='--', label=f'Euler, dt={dt_small:.2f}')
    ax.set_xlabel('Loss', fontsize=15)
    ax.set_ylabel('Error (%)', fontsize=15)
    ax.legend()
    return fig, ax


# initialize model
model = HamiltonianMappingModel(targetPotential=isoDiskPotential, input_dim=2, hidden_dim=128, num_layers=64)
model.flow = model.flow.double()

# generate training data
n_actions = 20
n_angles = 1000
r_min = 0.25
r_max = 1.25
omega_guess = 1
training_data, aa_guess = generate_sho_orbits(n_actions, omega=omega_guess, t_end=2*np.pi, n_steps=n_angles, r_bounds=[r_min,r_max])
aa_guess = aa_guess.to(torch.float64)

# train model 50 steps at a time

steps = 100
sets = 40
stepsizes = np.logspace(-2, 0, 10)
t_end = 10

dt_big = t_end/20
rk4_error_big_list = np.zeros(sets)
euler_error_big_list = np.zeros(sets)
rk4_std_big_list = np.zeros(sets)
euler_std_big_list = np.zeros(sets)

dt_small = t_end/40
rk4_error_small_list = np.zeros(sets)
euler_error_small_list = np.zeros(sets)
rk4_std_small_list = np.zeros(sets)
euler_std_small_list = np.zeros(sets)


ps_ = torch.tensor([1.0, 0], requires_grad=True).to(torch.float64)
H0_ = H(ps_, model.targetPotential)
for i in tqdm(range(sets)):
    print(f"Training set {i+1} of {sets}")
    model.train(training_data.to(torch.float64), steps, lr=1e-4)
    if i == 5:
        path = '/Users/gabrielpfaffman/Repos/orbitflows/examples/integration_tests/suite/imgs/poorly_trained/'
        # make plots...
        print('Generating poor training plots...')
        fig_step_1, ax_step_1 = step_size_cmap(model, stepsizes)
        fig_step_1.savefig(path+"stepsize_cmap_poor.png", dpi=300)
        fig_orbits_1, ax_orbits_1 = example_orbits_and_derivatives(model, training_data, n_grid_points=100)
        fig_orbits_1.savefig(path+"derivatives_poor.png", dpi=300)
    
    # collect percent error data for rk4 and euler at different step sizes
    aa_ = model.ps_to_aa(ps_)
    aa_rk4_big = model.integrate(aa_, 20, t_end, correction_step_size=None, correction=rk4, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
    aa_euler_big = model.integrate(aa_, 20, t_end, correction_step_size=None, correction=eulerstep, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
    euler_error_big_list[i] = np.mean(100 * (np.abs(model.hamiltonian(aa_euler_big).detach() - H0_.detach()) /  H0_.detach()).numpy())
    euler_std_big_list[i] = np.std(model.hamiltonian(aa_euler_big).detach().numpy())
    rk4_error_big_list[i] = np.mean(100 * (np.abs(model.hamiltonian(aa_rk4_big).detach() - H0_.detach()) /H0_.detach()).numpy())
    rk4_std_big_list[i]= np.std(model.hamiltonian(aa_rk4_big).detach().numpy())

    aa_rk4_small = model.integrate(aa_, 40, t_end, correction_step_size=None, correction=rk4, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
    aa_euler_small = model.integrate(aa_, 40, t_end, correction_step_size=None, correction=eulerstep, hamiltonian_tilde=h_fixed_angle).to(torch.float64)
    euler_error_small_list[i] = np.mean(100 * (np.abs(model.hamiltonian(aa_euler_small).detach() - H0_.detach()) /  H0_.detach()).numpy())
    euler_std_small_list[i] = np.std(model.hamiltonian(aa_euler_small).detach().numpy())
    rk4_error_small_list[i] = np.mean(100 * (np.abs(model.hamiltonian(aa_rk4_small).detach() - H0_.detach()) /H0_.detach()).numpy())
    rk4_std_small_list[i]= np.std(model.hamiltonian(aa_rk4_small).detach().numpy())

print(f"Training complete. Creating final plots...")
path = "/Users/gabrielpfaffman/Repos/orbitflows/examples/integration_tests/suite/imgs/well_trained/"
# Final plots
fig_step_2, ax_step_2 = step_size_cmap(model, stepsizes)
fig_step_2.savefig(path+"stepsize_cmap_well.png", dpi=300)

fig_orbits_2, ax_orbits_2 = example_orbits_and_derivatives(model, training_data, n_grid_points=100)
fig_orbits_2.savefig(path+"derivates_well.png", dpi=300)

path = "/Users/gabrielpfaffman/Repos/orbitflows/examples/integration_tests/suite/imgs/"
# plot the loss vs epoch
fig_loss, ax_loss = plt.subplots(1, 1, figsize=(7, 5))
ax_loss.plot(model.loss_list) # loss vs epoch
ax_loss.set_xlabel('epochs', fontsize=15)
ax_loss.set_ylabel('loss', fontsize=15)
fig_loss.savefig(path+"model_loss.png", dpi=300)

# plot the Hamiltonian before and after training
fig_H, ax_H = plt.subplots(1, 1, figsize=(7, 5))
ax_H.scatter(aa_guess[...,0], H(training_data, model.targetPotential).detach(), label='pre-training')
ax_H.scatter(aa_guess[...,0], model.hamiltonian(aa_guess).detach(), label='post-training')
ax_H.legend()
ax_H.set_xlabel('angle', fontsize=20)
ax_H.set_ylabel('energy', fontsize=20)
fig_H.savefig(path+"NF_H.png", dpi=300)

# error vs loss plot
fig_err_vs_loss, ax_err_vs_loss = error_vs_loss_plot(rk4_error_big_list, euler_error_big_list, rk4_error_small_list, euler_error_small_list, model.loss_list, dt_big, dt_small, stride=steps)
fig_err_vs_loss.savefig(path+"error_vs_loss.png", dpi=300)