'''
Training Functions for training sets with known action-angle
transformation.
'''

#from .context import polar_cartesian as pc
from .. import cartesian_to_polar, polar_to_cartesian
#from pc import cartesian_to_polar, polar_to_cartesian


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation, ArtistAnimation
from .. import H
import os
import json

# Get absolute path for the outputs directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../trained_models"))



def regular_train(
        x, 
        aa, 
        flow, 
        training_steps, 
        potential=None,
        plot=False, 
        plotcolor=False,
        savePlot=False,
        filename=None,
        anim=False,
        output_freq=50,
        batch_size=100, 
        lr=1e-3,
        duration=10,
        saveModel=False,
        training_data_name=None,
        gpu=None,
        ham_funct=H,
        *args,
        **kwargs
):
    

    '''
    Training when training set includes a guess for the correct
    action-angle coordinates and their associated correct action-angle 
    coordinates. Loss is MSE calculated in the cartesian Action-Angle basis.

    Params:
    ------
    x : input data [theta_guess, J_guess]
    aa : correct action angles [theta, J]
    flow : the normalizing flow model
    training_steps : number of training iterations
    batch_size : number of samples per batch
    gpu : None to run on cpu, device name as a string otherwise
    '''
    # change this in favor of just putting the tensors on the same device as the flow
    if gpu != None:
        device = gpu

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    

    if anim == True:
        figanim, axanim = plt.subplots(1, 3, figsize = (18, 6), sharex=False, sharey=False)
        spacer = 0.05
        axanim[0].set_xlim(-spacer, 2*np.pi+spacer)
        axanim[0].set_ylim(-0.05, 0.35)
        axanim[0].set_xlabel('$\\theta$')
        axanim[0].set_ylabel('J')
        axanim[0].set_title('True')

        axanim[1].set_xlim(-spacer, 2*np.pi+spacer)
        axanim[1].set_ylim(-0.05, 0.35)
        axanim[1].set_xlabel('$\\theta$')
        axanim[1].set_ylabel('J')
        axanim[1].set_title('NF Prediction')

        axanim[2].set_yscale('log')
        axanim[2].set_ylabel('loss')
        axanim[2].set_xlabel('epoch')

        ims = []

    # Create a DataLoader for batching
    dataset = TensorDataset(x, aa)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # For storing the Jacobian log-determinant at each step (if ded)
    jac_list = np.zeros(training_steps)
    plot_inds = np.random.randint(0, len(x), 10)
    loss_list = []  # List to record loss at each epoch

    for epoch in tqdm(range(training_steps)):
        for batch_idx, (batch_x, batch_aa) in enumerate(dataloader):
           
            if gpu != None:
                batch_x = batch_x.to(device)
                batch_aa = batch_aa.to(device)

            # Forward pass through the flow
            aa_i = flow(batch_x)
                
            # Compute the loss
            loss = F.mse_loss(aa_i, batch_aa) #+ torch.abs(torch.var(aa_i[:,0]) - torch.var(batch_aa[:,0]))

            # Backpropagation
            #optimizer.zero_grad()
            for param in flow.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

        # Record the loss
        loss_list.append(loss.item())

        # Logging and visualization
        if epoch % output_freq == 0:
            tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
                
            # Optional visualization for debugging
            if plot == True or anim == True:
                with torch.no_grad():
                    if gpu != None:
                        predicted_aa = flow(x.to(device))
                    else:
                        predicted_aa = flow(x)
            if plot == True:
                fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
                fig.suptitle(f'Epoch {epoch}; Action Angles', fontsize=16)
                if plotcolor == True:
                    ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = (ham_funct(*args, **kwargs)).T, cmap='inferno', s=1, alpha=1, label='True')
                    ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=(ham_funct(*args, **kwargs)).T, cmap='inferno', alpha=1, s=1, label='Predicted')
                    if savePlot == True:
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
            if anim == True:
                im1 = axanim[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = ham_funct(*args, **kwargs).T, cmap='inferno', s=1, alpha=1, label='True')
                im2 = axanim[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=ham_funct(*args, **kwargs).T, cmap='inferno', alpha=1, s=1, label='Predicted')
                im3, = axanim[2].plot(np.arange(0, epoch+1), np.array(loss_list), c='k')
                ims.append([im1, im2, im3])
                

    if saveModel == True:
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        # make a configuration file
        # Save the model state dictionary
        model_path = f'{model_dir}/{filename}.pt'
        torch.save(flow.state_dict(), model_path)

        # Create a configuration dictionary
        config = {
            "model_name": str(type(flow).__name__),
            "training_steps": training_steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "potential": str(potential) if potential != None else None,
            "potential_values": vars(potential) if potential != None else None,
            "filename": filename,
            "output_frequency": output_freq,
            "optimizer": "Adam",
            "num_layers": flow.num_layers,
            "hidden_dim": flow.hidden_dim,
            "input_dim": flow.input_dim,
            "training_data_name": training_data_name
        }

        # Save the configuration as a JSON file
        config_path = f'{model_dir}/{filename}_config.json'
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Configuration saved to {config_path}")
        torch.save(flow.state_dict(), f'{model_dir}/{filename}.pt')
        # save the loss list
        np.save(f'{model_dir}/{filename}_loss.npy', np.array(loss_list))

    if anim == True:
        if saveModel != True:
            model_dir = os.path.join(output_dir, filename)
            os.makedirs(model_dir, exist_ok=True)
        total_frames = epoch // output_freq
        interval = (duration * 1000) / total_frames
        
        anim = ArtistAnimation(figanim, ims, interval=interval, blit=True)        
        anim.save(f'{model_dir}/{filename}.mp4', writer='ffmpeg')
                    
                    # fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
                    # fig.suptitle(f'Epoch {epoch}; Transformed Points for Loss Calculation', fontsize=16)
                    # if plot == True:
                    #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
                    #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
                    # else:
                    #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = 'k', s=5, cmap='inferno')
                    #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = 'k', s=5, cmap='inferno')
                
    return loss_list


def polar_train(
        x, 
        aa, 
        flow, 
        training_steps, 
        potential=None,
        plot=False, 
        plotcolor=False,
        savePlot=False,
        filename=None,
        anim=False,
        output_freq=50,
        batch_size=100, 
        lr=1e-3,
        duration=10,
        saveModel=False,
        training_data_name=None,
        ham_funct=H,
        *args,
        **kwargs
):
    

    '''
    Training when training set includes a guess for the correct
    action-angle coordinates and their associated correct action-angle 
    coordinates. Loss is MSE calculated in the cartesian Action-Angle basis.

    Params:
    ------
    x : input data [theta_guess, J_guess]
    aa : correct action angles [theta, J]
    flow : the normalizing flow model
    training_steps : number of training iterations
    batch_size : number of samples per batch
    gpu : None to run on cpu, device name as a string otherwise
    '''
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    gpu_available = False

    if torch.cuda.is_available() == True:
        device = 'cuda'
        flow.to(device)
        x.to(device)
        aa.to(device)
        gpu_available = True
        print('Added flow to gpu.')
    else:
        print('gpu unavailable')
        print(torch.cuda.is_available())

    if anim == True:
        figanim, axanim = plt.subplots(1, 3, figsize = (18, 6), sharex=False, sharey=False)
        spacer = 0.05
        axanim[0].set_xlim(-spacer, 2*np.pi+spacer)
        axanim[0].set_ylim(-0.05, 0.35)
        axanim[0].set_xlabel('$\\theta$')
        axanim[0].set_ylabel('J')
        axanim[0].set_title('True')

        axanim[1].set_xlim(-spacer, 2*np.pi+spacer)
        axanim[1].set_ylim(-0.05, 0.35)
        axanim[1].set_xlabel('$\\theta$')
        axanim[1].set_ylabel('J')
        axanim[1].set_title('NF Prediction')

        axanim[2].set_yscale('log')
        axanim[2].set_ylabel('loss')
        axanim[2].set_xlabel('epoch')

        ims = []

    # Create a DataLoader for batching
    dataset = TensorDataset(x, aa)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True).to(device)

    # For storing the Jacobian log-determinant at each step (if ded)
    jac_list = np.zeros(training_steps)
    plot_inds = np.random.randint(0, len(x), 10)
    loss_list = []  # List to record loss at each epoch

    for epoch in tqdm(range(training_steps)):
        for batch_idx, (batch_x, batch_aa) in enumerate(dataloader):
            # if  gpu_available:
            #     batch_x = batch_x.to(device)
            #     batch_aa = batch_aa.to(device)

            # Forward pass through the flow
            aa_i = flow(batch_x)
                
            # Compute the loss
            loss = F.mse_loss(polar_to_cartesian(aa_i), polar_to_cartesian(batch_aa)) #+ torch.abs(torch.var(aa_i[:,0]) - torch.var(batch_aa[:,0]))

            # Backpropagation
            #optimizer.zero_grad()
            for param in flow.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

        # Record the loss
        loss_list.append(loss.item())

        # Logging and visualization
        if epoch % output_freq == 0:
            tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
                
            # Optional visualization for debugging
            if plot == True or anim == True:
                with torch.no_grad():
                    if gpu_available == True:
                        predicted_aa = flow(x.to(device))
                    else:
                        predicted_aa = flow(x)
            if plot == True:
                fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
                fig.suptitle(f'Epoch {epoch}; Action Angles', fontsize=16)
                if plotcolor == True:
                    ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = (ham_funct(*args, **kwargs)).T, cmap='inferno', s=1, alpha=1, label='True')
                    ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=(ham_funct(*args, **kwargs)).T, cmap='inferno', alpha=1, s=1, label='Predicted')
                    if savePlot == True:
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
            if anim == True:
                im1 = axanim[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = ham_funct(*args, **kwargs).T, cmap='inferno', s=1, alpha=1, label='True')
                im2 = axanim[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=ham_funct(*args, **kwargs).T, cmap='inferno', alpha=1, s=1, label='Predicted')
                im3, = axanim[2].plot(np.arange(0, epoch+1), np.array(loss_list), c='k')
                ims.append([im1, im2, im3])
                

    if saveModel == True:
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        # make a configuration file
        # Save the model state dictionary
        model_path = f'{model_dir}/{filename}.pt'
        torch.save(flow.state_dict(), model_path)

        # Create a configuration dictionary
        config = {
            "model_name": type(flow).__name__,
            "training_steps": int(training_steps),
            "batch_size": int(batch_size),
            "learning_rate": lr,  # Assuming lr is a float
            "potential": str(potential) if potential is not None else None,
            "potential_values": vars(potential) if potential is not None else None,
            "filename": filename,
            "output_frequency": int(output_freq),
            "optimizer": "Adam",
            "num_layers": int(flow.num_layers),
            "hidden_dim": int(flow.hidden_dim),
            "input_dim": int(flow.input_dim),
            "training_data_name": training_data_name,
            "gpu": gpu_available
            }

        # Save the configuration as a JSON file
        config_path = f'{model_dir}/{filename}_config.json'
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Configuration saved to {config_path}")
        torch.save(flow.state_dict(), f'{model_dir}/{filename}.pt')
        # save the loss list
        np.save(f'{model_dir}/{filename}_loss.npy', np.array(loss_list))

    if anim == True:
        if saveModel != True:
            model_dir = os.path.join(output_dir, filename)
            os.makedirs(model_dir, exist_ok=True)
        total_frames = epoch // output_freq
        interval = (duration * 1000) / total_frames
        
        anim = ArtistAnimation(figanim, ims, interval=interval, blit=True)        
        anim.save(f'{model_dir}/{filename}.mp4', writer='ffmpeg')
                    
                    # fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
                    # fig.suptitle(f'Epoch {epoch}; Transformed Points for Loss Calculation', fontsize=16)
                    # if plot == True:
                    #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
                    #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
                    # else:
                    #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = 'k', s=5, cmap='inferno')
                    #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = 'k', s=5, cmap='inferno')
                
    return loss_list


# def polar_train(
#         x, 
#         aa, 
#         flow, 
#         training_steps, 
#         potential=None,
#         plot=False, 
#         plotcolor=False,
#         savePlot=False,
#         filename=None,
#         anim=False,
#         output_freq=50,
#         batch_size=100, 
#         lr=1e-3,
#         duration=10,
#         saveModel=False,
#         training_data_name=None
# ):
    

#     '''
#     Training when training set includes a guess for the correct
#     action-angle coordinates and their associated correct action-angle 
#     coordinates. Loss is MSE calculated in the cartesian Action-Angle basis.

#     Params:
#     ------
#     x : input data [theta_guess, J_guess]
#     aa : correct action angles [theta, J]
#     flow : the normalizing flow model
#     training_steps : number of training iterations
#     batch_size : number of samples per batch
#     '''
#     optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    

#     if anim == True:
#         figanim, axanim = plt.subplots(1, 3, figsize = (18, 6), sharex=False, sharey=False)
#         spacer = 0.05
#         axanim[0].set_xlim(-spacer, 2*np.pi+spacer)
#         axanim[0].set_ylim(-0.05, 0.35)
#         axanim[0].set_xlabel('$\\theta$')
#         axanim[0].set_ylabel('J')
#         axanim[0].set_title('True')

#         axanim[1].set_xlim(-spacer, 2*np.pi+spacer)
#         axanim[1].set_ylim(-0.05, 0.35)
#         axanim[1].set_xlabel('$\\theta$')
#         axanim[1].set_ylabel('J')
#         axanim[1].set_title('NF Prediction')

#         axanim[2].set_yscale('log')
#         axanim[2].set_ylabel('loss')
#         axanim[2].set_xlabel('epoch')

#         ims = []

#     # Create a DataLoader for batching
#     dataset = TensorDataset(x, aa)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # For storing the Jacobian log-determinant at each step (if ded)
#     jac_list = np.zeros(training_steps)
#     plot_inds = np.random.randint(0, len(x), 10)
#     loss_list = []  # List to record loss at each epoch

#     for epoch in tqdm(range(training_steps)):
#         for batch_idx, (batch_x, batch_aa) in enumerate(dataloader):
#             # Forward pass through the flow
#             aa_i = flow(batch_x)
                
#             # Compute the loss
#             loss = F.mse_loss(polar_to_cartesian(aa_i), polar_to_cartesian(batch_aa)) #+ torch.abs(torch.var(aa_i[:,0]) - torch.var(batch_aa[:,0]))

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # Record the loss
#         loss_list.append(loss.item())

#         # Logging and visualization
#         if epoch % output_freq == 0:
#             tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
                
#             # Optional visualization for debugging
#             if plot == True or anim == True:
#                 with torch.no_grad():
#                     predicted_aa = flow(x)
#             if plot == True:
#                 fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
#                 fig.suptitle(f'Epoch {epoch}; Action Angles', fontsize=16)
#                 if plotcolor == True:
#                     ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = H(z, potential).T, cmap='inferno', s=1, alpha=1, label='True')
#                     ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=H(z, potential).T, cmap='inferno', alpha=1, s=1, label='Predicted')
#                     if savePlot == True:
#                         plt.savefig(f'{output_dir}/{filename}_{epoch}.png')
#                     else:
#                         plt.show()
#                 else:
#                     ax[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = 'k', cmap='inferno', s=20, alpha=0.2, label='True')
#                     ax[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c='k', cmap='inferno', alpha=1, s=1, label='Predicted')
#                     if savePlot == True:
#                         plt.savefig(f'{output_dir}/{filename}_{epoch}.png')
#                     else:
#                         plt.show()
#             if anim == True:
#                 im1 = axanim[0].scatter(aa.cpu().numpy()[:, 0], aa.cpu().numpy()[:, 1], c = H(z, potential).T, cmap='inferno', s=1, alpha=1, label='True')
#                 im2 = axanim[1].scatter(predicted_aa.cpu().numpy()[:, 0], predicted_aa.cpu().numpy()[:, 1], c=H(z, potential).T, cmap='inferno', alpha=1, s=1, label='Predicted')
#                 im3, = axanim[2].plot(np.arange(0, epoch+1), np.array(loss_list), c='k')
#                 ims.append([im1, im2, im3])
                

#     if saveModel == True:
#         model_dir = os.path.join(output_dir, filename)
#         os.makedirs(model_dir, exist_ok=True)
#         # make a configuration file
#         # Save the model state dictionary
#         model_path = f'{model_dir}/{filename}.pt'
#         torch.save(flow.state_dict(), model_path)

#         # Create a configuration dictionary
#         config = {
#             "model_name": type(flow).__name__,
#             "training_steps": training_steps,
#             "batch_size": batch_size,
#             "learning_rate": lr,
#             "potential": str(potential) if potential else None,
#             "potential_values": vars(potential),
#             "filename": filename,
#             "output_frequency": output_freq,
#             "optimizer": "Adam",
#             "num_layers": flow.num_layers,
#             "hidden_dim": flow.hidden_dim,
#             "input_dim": flow.input_dim,
#             "training_data_name": training_data_name
#         }

#         # Save the configuration as a JSON file
#         config_path = f'{model_dir}/{filename}_config.json'
#         with open(config_path, 'w') as config_file:
#             json.dump(config, config_file, indent=4)

#         print(f"Model saved to {model_path}")
#         print(f"Configuration saved to {config_path}")
#         torch.save(flow.state_dict(), f'{model_dir}/{filename}.pt')
#         # save the loss list
#         np.save(f'{model_dir}/{filename}_loss.npy', np.array(loss_list))

#     if anim == True:
#         if saveModel != True:
#             model_dir = os.path.join(output_dir, filename)
#             os.makedirs(model_dir, exist_ok=True)
#         total_frames = epoch // output_freq
#         interval = (duration * 1000) / total_frames
        
#         anim = ArtistAnimation(figanim, ims, interval=interval, blit=True)        
#         anim.save(f'{model_dir}/{filename}.mp4', writer='ffmpeg')
                    
#                     # fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharex=True, sharey=True)
#                     # fig.suptitle(f'Epoch {epoch}; Transformed Points for Loss Calculation', fontsize=16)
#                     # if plot == True:
#                     #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
#                     #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = H(z, potential).T, s=5, cmap='inferno')
#                     # else:
#                     #     ax[0].scatter(polar_to_cartesian(aa)[:,0], polar_to_cartesian(aa)[:,1], c = 'k', s=5, cmap='inferno')
#                     #     ax[1].scatter(polar_to_cartesian(predicted_aa)[:,0], polar_to_cartesian(predicted_aa)[:,1], c = 'k', s=5, cmap='inferno')
                
#     return loss_list