from .potentials import *
from ..flow import GradientBasedConditioner, NNConditioner, SimpleNNConditioner, SymplecticCouplingLayer

potential_key_mappings = {
    'isoDiskPotential': isoDiskPotential,
    'MWPotential2014_1D': MWPotential2014_1D
}
potential_function_mappings = {
    isoDiskPotential: 'isoDiskPotential',
}

conditioner_key_mappings = {
    'GradientBasedConditioner': GradientBasedConditioner,
    'NNConditioner': NNConditioner,
    'SimpleNNConditioner': SimpleNNConditioner,
}

conditioner_function_mappings = {
    GradientBasedConditioner: 'GradientBasedConditioner',
    NNConditioner: 'NNConditioner',
    SimpleNNConditioner: 'SimpleNNConditioner',
}

layer_key_mappings = {
    'SymplecticCouplingLayer': SymplecticCouplingLayer,
}

layer_function_mappings = {
    SymplecticCouplingLayer: 'SymplecticCouplingLayer',
}