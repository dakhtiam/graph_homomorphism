"""
Data transforms for Embedding classes
"""
# imports
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

import numpy as np

import networkx as nx
import graph_encoding.encoding as encoding


from typing import Set
import itertools
from functools import partial


###### Setting up the embeding ########
class SubgraphTransform(BaseTransform):
    def __init__(self):
        pass


def add_testgraphs(self, data: encoding.grandEmbedding,
                   limit_vertex=None,
                   n_trees=2, limit_trees=10000,
                   n_cycles=3, limit_cycles=10000,
                   n_cliques=4, limit_cliques=100):
    '''
    Clearing all testgraphs in data and adding to the testgraph set:
    - a single vertex
    - all non-isomorphic trees of sizes in range(2, n_trees) (default default no graphs)
    - cycles of length range(3, n_cyces) (default no graphs)
    - cliques of sizes in range(4, n_cliques) (default no graphs)
    '''

    data.clear_all_testgraphs()
    data.add_single_vertex(limit=limit_vertex)
    data.add_trees(stop=n_trees, limit=limit_trees)
    data.add_cycles(stop=n_cycles, limit=limit_cycles)
    data.add_cliques(stop=n_cliques, limit=limit_cliques)


def grand_transform(data: Data, limit_vertex=None,
                    n_trees=2, limit_trees=10000,
                    n_cycles=3, limit_cycles=10000,
                    n_cliques=4, limit_cliques=100):
    '''
    Taking a torch_geometric.data object and returning 
    a grandEmbeddig with test graphs given by trees, cycles, cliques 
    as specified in add_testgraphs()
    '''
    encoded_data = encoding.grandEmbedding(data)
    add_testgraphs(encoded_data, limit_vertex=None,
                   n_trees=2, limit_trees=10000,
                   n_cycles=3, limit_cycles=10000,
                   n_cliques=4, limit_cliques=100)
    return encoded_data


if __name__ == "__main__":
    import doctest
    doctest.testmod()
