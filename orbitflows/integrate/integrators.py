'''Implementation of integrators using the ML transformations.'''


def integrate_with_correction(aa0, n_steps, t_end):
    '''Routine to integrate orbits in AA space and update frequencies regularly, with euler step correction
    
    Parameters
    ----------
    aa0 : torch.tensor
        initial action-angle variables
    steps_aa : int
        number of steps in action-angle space between frequency updates'''
    
    delta_t = torch.tensor(t_end/n_steps)
    theta_list = torch.zeros(n_steps)
    j_list = torch.zeros(n_steps)
    
    theta0 = aa0[0]
    j0 = aa0[1]
    freq = model.frequency(theta_set, j0)
    

    for i, t in enumerate(tqdm(np.linspace(0, t_end, n_steps))):
        # evolve in action-angle space
        theta_half = theta0 + freq*delta_t/2 # drift 

        # transform to phase space
        ps_int = aAHI(j0, theta_half) # intermediate phase space position (guess)
        ps_nf = model.flow(ps_int.T) # nf_solution
        p0 = ps_nf[1]
        q0 = ps_nf[0]

        # Euler step
        p = p0 - delta_t * torch.autograd.grad(h_error(q0, p0), q0)[0]
        q = q0 + delta_t * torch.autograd.grad(h_error(q0, p0), p0)[0]

        # convert back to action-angle space
        ps_int = torch.stack((q, p))
        ps_nf = model.flow.inverse(ps_int.T)
        act, _, angle = aAH._actionsFreqsAngles(*ps_nf.T)
        j = torch.tensor(act, requires_grad=True)
        theta_half_2 = torch.tensor(angle, requires_grad=True)
        freq = torch.autograd.grad(hamiltonian_J(J), J)[0]
        theta = theta_half_2 + freq*delta_t/2 # drift

        theta_list[i] = theta
        j_list[i] = j
        theta0 = theta
        j0 = j
        #print(theta0, J0)
    return J_list, theta_list