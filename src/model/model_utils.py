import copy

from torch import nn


def get_clones(module: nn.Module, num_layers: int):
    """
    Clone layer module to generate multiple layers

    Args:
        module: encoder or decoder layer
        num_layers: number of layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
