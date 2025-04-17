
'''
Plotting functions
'''

import numpy as np
import matplotlib.pyplot as plt
from .hamiltonians import H
import os

def plot_aa(n_orbits, aa, t_end, n_steps, t_start=0, color=H, color_kwargs={}, plot_kwargs={'s':5, 'cmap':'inferno'}):
    '''
    color (function or string) : color to plot by
    
    '''

    t_ls = np.linspace(t_start, t_end, n_steps)
    fig, ax = plt.subplots(2, 1, figsize = (12, 8), sharex=True, sharey=False)
    if type(color) == str:
        ax[0].scatter(np.array([t_ls for i in np.arange(0, n_orbits)]), aa[:,0], c = 'k', **plot_kwargs)
        ax[1].scatter(np.array([t_ls for i in np.arange(0, n_orbits)]), aa[:,1], c = 'k', **plot_kwargs)
    else:
        ax[0].scatter(np.array([t_ls for i in np.arange(0, n_orbits)]), aa[:,0], c = color(**color_kwargs), **plot_kwargs)
        ax[1].scatter(np.array([t_ls for i in np.arange(0, n_orbits)]), aa[:,1], c = color(**color_kwargs), **plot_kwargs)

    ax[0].set_xlabel('t', fontsize=11)
    ax[0].set_ylabel('$\\theta$', fontsize=11)

    ax[1].set_title('Analytic (Training Set)', fontsize=12)
    ax[1].set_xlabel('t', fontsize=11)
    ax[1].set_ylabel('J', fontsize=11)

def plot_ps(n_orbits, z, t_end, n_steps, t_start=0, color=H, color_kwargs={}, plot_kwargs={'s':5, 'cmap':'inferno'}):
    '''
    color (function or string) : color to plot by
    
    '''
    t_ls = np.linspace(t_start, t_end, n_steps)
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    if type(color) == str:
        ax.scatter(z[:,0], z[:,1], c = 'k', **plot_kwargs)
    else:
        ax.scatter(z[:,0], z[:,1], c = color(**color_kwargs), **plot_kwargs)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('v', fontsize=11)

    ax.set_title('Phase Space', fontsize=12)

def training_plot_known(aa, predicted_aa, epoch, savePlot=True, plotColor=False, ham_funct=H, h_params={}, save_details={'output_dir' : os.path.abspath(os.path.join(os.path.dirname(__file__), "../../trained_models")), 'filename':'training_plot'}):
    '''
    Plot training progress for cases with
    known action-angle solution
    '''
    fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
    fig.suptitle(f'Epoch {epoch}; Action Angles', fontsize=16)
    if plotColor == True:
        ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = (ham_funct(**h_params)).T, cmap='inferno', s=1, alpha=1, label='True')
        ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=(ham_funct(*args, **kwargs)).T, cmap='inferno', alpha=1, s=1, label='Predicted')
        if savePlot == True:
            output_dir = save_details['output_dir']
            filename = save_details['filename']
            plt.savefig(f'{output_dir}/{filename}_{epoch}.png')
        else:
            plt.show()
    else:
        ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = 'k', cmap='inferno', s=1, alpha=1, label='True')
        ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c='k', cmap='inferno', alpha=1, s=1, label='Predicted')
        if savePlot == True:
            plt.savefig(f'{output_dir}/{filename}_{epoch}.png')
        else:
            plt.show()