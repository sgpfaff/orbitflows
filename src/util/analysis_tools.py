import torch

def mean_along_orbs(value_list):
    return torch.mean(value_list, axis=1).unsqueeze(1)

def diff_from_mean_along_orbs(value_list):
    return value_list - mean_along_orbs(value_list)

def percent_error_along_orbs(value_list):
    return torch.log10(torch.abs(diff_from_mean_along_orbs(value_list) / mean_along_orbs(value_list)))

def max_error_along_orbs(value_list):
    return torch.max(percent_error_along_orbs(value_list), axis=1).values

def mean_error_along_orbs(value_list):
    """
    Computes the mean error of the Hamiltonian along the orbit.
    """
    return mean_along_orbs(percent_error_along_orbs(value_list))
