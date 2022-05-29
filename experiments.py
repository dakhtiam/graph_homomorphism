"""
Set up measurements for different pattern graphs
"""
# imports
import string
from tokenize import String
import numpy as np
#import torch
#from torch_geometric.data import Data
#from torch_geometric.transforms import BaseTransform
#from torch_geometric.datasets import TUDataset, ZINC
#from ogb.graphproppred import PygGraphPropPredDataset

import networkx as nx
from lib.graph_encoding.encoding import grandEmbedding

from typing import Dict
import itertools
from functools import partial
from dataclasses import dataclass

#from tqdm import tqdm

# ogb
#from ogb.graphproppred import Evaluator


###### Setting up the embeding ########
class ExperimentEmbedding(grandEmbedding):
    def __init__(self, data):
        super().__init__(data)

    def set_testgraphs(self, limit_vertex=None,
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
        self.clear_all_testgraphs()
        self.add_single_vertex(limit=limit_vertex)
        self.add_trees(stop=n_trees, limit=limit_trees)
        self.add_cycles(stop=n_cycles, limit=limit_cycles)
        self.add_cliques(stop=n_cliques, limit=limit_cliques)


###### Save processed data ########

####### load data ########

######### experiment pipelines #######
@dataclass
class Result:
    X: np.ndarray
    y: np.ndarray
    nums: Dict[str, int]


class patternExperiment():
    def __init__(self, dataset, folder_name,
                 encoder_name, limit_vertex=None,
                 n_trees=2, limit_trees=10000,
                 n_cycles=3, limit_cycles=10000,
                 n_cliques=4, limit_cliques=100):
        self.dataset = dataset
        self.encoded_dataset = None
        self.dataset_name = f'{dataset}'
        self.folder_name = folder_name
        self.encoder_name = encoder_name

        self.limit_vertex = limit_vertex
        self.n_trees = n_trees
        self.limit_trees = limit_trees
        self.n_cycles = n_cycles
        self.limit_cycles = limit_cycles
        self.n_cliques = n_cliques
        self.limit_cliques = limit_cliques

    def __init_dataset(self):
        self.encoded_dataset = [
            ExperimentEmbedding(data) for data in self.dataset]

    def __encoder_dataset(self, limit_vertex=None,
                          n_trees=2, limit_trees=10000,
                          n_cycles=3, limit_cycles=10000,
                          n_cliques=4, limit_cliques=100):
        # labels
        if self.dataset_name == 'PygGraphPropPredDataset(41127)':
            y = np.array([data.pyg_graph().y[0, 0].detach().numpy()
                          for data in self.encoded_dataset])
        else:
            y = np.array([data.pyg_graph().y.detach().numpy()
                          for data in self.encoded_dataset])

        if self.encoder_name == 'ghc_aug':
            def pure_encoder(x): return x.ghc_encoder(format='numpy')
        elif self.encoder_name == 'lagrangian_aug':
            def pure_encoder(x): return x.lagrangian_encoder(format='numpy')

        def num_enc(x): return x.num_encoder(format='numpy')

        def encoder(x): return np.concatenate(
            (pure_encoder(x), num_enc(x)), axis=0)

        # feature vector
        def set_graphs(x): return x.set_testgraphs(limit_vertex,
                                                   n_trees, limit_trees,
                                                   n_cycles, limit_cycles,
                                                   n_cliques, limit_cliques)
        __add_to_Dataset = [set_graphs(data)
                            for data in self.encoded_dataset]
        X = np.array([encoder(data)
                      for data in self.encoded_dataset])

        return X, y

    def __save_data(self, i, X, y, nums):
        np.save(self.folder_name + '/'+self.encoder_name+'/' + f'{i}_X.npy', X)
        np.save(self.folder_name+'/'+self.encoder_name+'/' + f'{i}_y.npy', y)
        np.save(self.folder_name + '/'+self.encoder_name +
                '/' + f'{i}_nums.npy', nums, allow_pickle=True)

    def load_data(self):
        datadict = {}
        for i in range(self.num_of_experiments):
            X = np.load(self.folder_name + '/' +
                        self.encoder_name+'/' + f'{i}_X.npy')
            y = np.load(self.folder_name+'/' +
                        self.encoder_name+'/' + f'{i}_y.npy')
            nums = np.load(self.folder_name + '/'+self.encoder_name +
                           '/' + f'{i}_nums.npy', allow_pickle=True).item()
            result = Result(X, y, nums)
            datadict[i] = result
        return datadict

    def __evaluation(self):
        pass

    def run(self):
        self.__init_dataset()

        i = 0
        for n_cliques in range(4, self.n_cliques):
            for n_cycles in range(3, self.n_cycles):
                for n_trees in range(2, self.n_trees):
                    X, y = self.__encoder_dataset(limit_vertex=self.limit_vertex,
                                                  n_trees=n_trees, limit_trees=self.limit_trees,
                                                  n_cycles=n_cycles, limit_cycles=self.limit_cycles,
                                                  n_cliques=n_cliques, limit_cliques=self.limit_cliques)
                    nums = np.array({'n_trees': n_trees,
                                     'n_cycles': n_cycles, 'n_cliques': n_cliques})
                    self.__save_data(i, X, y, nums)
                    i += 1
        self.num_of_experiments = i


if __name__ == "__main__":
    import doctest
    doctest.testmod()
