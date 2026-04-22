from ..dynamics.potentials import *
from ..models.flow import (GradientBasedConditioner, NNConditioner, SimpleNNConditioner, 
                      SymplecticCouplingLayer, TorusSymplecticCouplingLayer, WrappedAnglesCouplingLayer)


potential_key_mappings = {
    'sho_potential': sho_potential,
    'isoDiskPotential': isoDiskPotential,
    'MWPotential2014_1D': MWPotential2014_1D
}
potential_function_mappings = {
    isoDiskPotential: 'isoDiskPotential',
    sho_potential: 'sho_potential',
    MWPotential2014_1D: 'MWPotential2014_1D',
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
    'TorusSymplecticCouplingLayer': TorusSymplecticCouplingLayer,
    'WrappedAnglesCouplingLayer': WrappedAnglesCouplingLayer,
}

layer_function_mappings = {
    SymplecticCouplingLayer: 'SymplecticCouplingLayer',
    TorusSymplecticCouplingLayer: 'TorusSymplecticCouplingLayer',
    WrappedAnglesCouplingLayer: 'WrappedAnglesCouplingLayer',
}

optimizer_key_mappings = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}

optimizer_function_mappings = {
    torch.optim.Adam: 'Adam',
    torch.optim.SGD: 'SGD',
    
}

scheduler_key_mappings = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

scheduler_function_mappings = {
    torch.optim.lr_scheduler.ReduceLROnPlateau: 'ReduceLROnPlateau',
}