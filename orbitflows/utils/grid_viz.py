'''script to animate transformation of grid and other useful plots'''
import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import matplotlib.animation as animation

def make_grid(plot_res=0.1, lim = 4, in_between=5):
    '''makes a grid of points in phase space
    
    NOTE: Outputs float64; May need to change this.
    '''
    res = int(in_between)
    n = int(2*lim*in_between/(plot_res))
    vel = torch.arange(-lim, lim, 0.1/in_between)
    pos = torch.arange(-lim, lim, 0.1/in_between)
    pos, vel = torch.meshgrid(pos, vel)
    return torch.stack([pos, vel], axis=-1).reshape(-1, 2).to(torch.float64), res, n

def transformed_grid_plot(model, plot_res=0.1, lim=4, show=True):
    '''
    Make plot of transformed phase space grid
    
    Parameters
    ----------
    model : orbitflows Model
        ML model of which you'd like to visualize the transformation
    '''
    ps, res, n = make_grid(plot_res=plot_res, lim=lim) # make ps grid
    ps_nf = model.flow(ps) # transform grid with ML model
    nf_plot = ps_nf.reshape(n, n, 2)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(nf_plot[...,0].cpu().detach()[:,::res], nf_plot[...,1].cpu().detach()[:,::res], c='k', alpha=0.5)
    ax.plot(nf_plot[...,0].cpu().detach().T[:,::res], nf_plot[...,1].cpu().detach().T[:,::res], c='k', alpha=0.5)

    ax.set_xlabel('position')
    ax.set_ylabel('momentum')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xticks(np.linspace(-1, 1, int(n/(res*lim))))
    ax.set_yticks(np.linspace(-1, 1, int(n/(res*lim))))
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(np.arange(-1, 1, plot_res))])
    ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(np.arange(-1, 1,plot_res))])
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    if show:
        plt.show()

    return fig, ax



def animate_transformation(
        model, 
        plot_res=0.1, 
        lim = 4, 
        in_between=5,
        orbits_ps = None
        ):
    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize empty plots for horizontal and vertical lines
    horizontal_lines, = ax.plot([], [], 'k', alpha=0.5)
    vertical_lines, = ax.plot([], [], 'k', alpha=0.5)

    # Set up the axis properties
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('position', fontsize=14)
    ax.set_ylabel('velocity', fontsize=14)
    ax.set_xticks(np.arange(-1, 1, plot_res))
    ax.set_yticks(np.arange(-1, 1, plot_res))
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(np.arange(-1, 1, plot_res))])
    ax.set_yticklabels([f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(np.arange(-1, 1, plot_res))])
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

    # Title with layer counter
    title = ax.set_title('Original Grid', fontsize=16)
    ps, res, n = make_grid(plot_res, lim, in_between)


    def init():
        horizontal_lines.set_data([], [])
        vertical_lines.set_data([], [])
        return horizontal_lines, vertical_lines, title

    def animate(i):
        if i == 0:
            # Original grid
            if orbits_ps != None:
                orbit_transformed = orbits_ps.clone()
            grid = ps.reshape(n, n, 2)
            title.set_text('Original Grid')
        else:
            # Apply the first i layers
            if orbits_ps != None:
                orbit_transformed = orbits_ps.clone()
            ps_transformed = ps.clone()
            for layer_idx in range(i):
                if orbits_ps != None:
                    orbit_transformed = model.flow.layers[layer_idx](orbit_transformed)
                ps_transformed = model.flow.layers[layer_idx](ps_transformed)
            grid = ps_transformed.reshape(n, n, 2)

        # Extract x and y coordinates for the horizontal and vertical grid lines
        h_x = grid[..., 0].cpu().detach().numpy()
        h_y = grid[..., 1].cpu().detach().numpy()
        v_x = h_x.T
        v_y = h_y.T
        
        # Update the data for the grid lines
        # Plot each horizontal line separately to prevent wrapping
        horizontal_lines.set_data([], [])  # Clear existing data
        vertical_lines.set_data([], [])    # Clear existing data
        
        # Clear previous lines
        for line in ax.get_lines():
            if line != horizontal_lines and line != vertical_lines:
                line.remove()
        
        # Draw each horizontal line (row)
        for j in np.arange(0, n, res):
            ax.plot(h_x[j], h_y[j], 'k', alpha=0.5)
        
        # Draw each vertical line (column)
        for j in np.arange(0, n, res):
            ax.plot(v_x[j], v_y[j], 'k', alpha=0.5)
        
         
        if i > 0:
            title.set_text(f'$\\mathbf{{\\psi_{{{i}}}}}\circ \cdot \cdot \cdot \circ \\psi_1(q, p)$')
        
        scatter = False
        if orbits_ps != None:
            if scatter:
                ax.scatter(orbit_transformed[...,0].cpu().detach(), orbit_transformed[...,1].cpu().detach(), c='cornflowerblue', s=10)
            else:
                for orb in orbit_transformed.reshape(orbits_ps.shape):
                    ax.plot(orb[:,0].cpu().detach(), orb[:,1].cpu().detach(), c='cornflowerblue', linewidth=3.5)
            
        return horizontal_lines, vertical_lines, title

    # Create the animation
    total_layers = len(model.flow.layers)
    ani = animation.FuncAnimation(fig, animate, frames=total_layers+1,
                                init_func=init, blit=True, interval=500)
    return ani

def display_animation(ani):
    '''
    Display animation in jupyter notebook.
    
    Doesn't work yet... Needs to be debugged
    '''
    HTML(ani.to_jshtml())

def save_animation(
        ani, 
        filename='grid_transformation', 
        fps=7, 
        bitrate=1800, 
        yourname=''):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps, metadata=dict(artist=yourname), bitrate=bitrate)
    output_file = filename + ".mp4"
    ani.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")