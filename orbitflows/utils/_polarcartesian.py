'''
Not implemented yet.

Coordinate transformation between polar and
cartesian coordinates.
'''

import torch

def cartesian_to_polar(z):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters:
    ----------
    z : torch.tensor
        Cartesian coordinates (x, y), shape = (n, 2). Where
        n is the number of points.

    Returns:
    --------
    torch.tensor
        Polar coordinates (theta, r), shape = (n, 2).
        where r is radius and theta (float or torch.Tensor) is angle in radians
    """
    x, y = z[:, 0], z[:, 1]
    r = torch.sqrt(x**2 + y**2)[:, None]
    theta = torch.atan2(y, x)[:, None]
    return torch.cat((theta, r), dim=-1)

def polar_to_cartesian(z):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters:
    ----------
    z : torch.tensor
        Polar coordinates (theta, r), shape = (n, 2). Where
        n is the number of points.

    Returns:
    --------
    torch.tensor
        Cartesian coordinates (x, y), shape = (n, 2).
        where x and y are the coordinates.
    """
    theta, r = z[:, 0], z[:, 1]
    x = (r * torch.cos(theta))[:, None]
    y = (r * torch.sin(theta))[:, None]
    return torch.cat((x, y), dim=-1)