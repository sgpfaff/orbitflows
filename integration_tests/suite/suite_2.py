
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

hidden_layers_list = [2, 10, 50, 100, 164]
number_of_layers_list = [2, 5, 10, 25, 64]

for 