"""
A mini-library to process graphs
"""
# imports
import torch
from torch_geometric.data import Data
import torch_geometric.utils as uts

import numpy as np
import networkx as nx
from grandiso import find_motifs

from typing import Set
import itertools
import functools

####### testgraph and testset classes ##########


class testGraph:
    '''
    A class object for a test graph with descriptive features
    '''

    def __init__(self, graph, graph_name=None, limit=None, format='networkx'):
        '''
         Parameters:
         graph: a graph in the format "format" (default is networkX)
         graph_name (String): the name of a graph 
         max_n (Int): a bound for the number of subgraphs taken
        '''
        self.name = graph_name
        self.__graph = graph
        self.bound = limit
        self.format = format

    def nx_graph(self):
        '''
         returns the graph in networkX format
        '''
        return self.__graph

    def pyg_graph(self):
        if self.format != 'networkx':
            raise Exception('Not implemented yet')
        return uts.convert.from_networkx(self.__graph)

    def draw(self):
        '''
         returns a drawing of the graph
        '''
        if self.format != 'networkx':
            return None
        return nx.draw(self.nx_graph())


# make a type alias for a set of test graphs:
testGraphSet = Set[testGraph]


####### generic Embedding classes ##########


class Embedding():
    '''
    A parent class for for handeling the embeddings from subgraph isomorphism. 
    A specific class using aparticular graph homomorphism computation algorithm or library 
    inherits this class
    '''

    def __init__(self, graph,
                 testgraphs: testGraphSet = {},
                 symmetry=False,
                 induced=False,
                 undirected='True',
                 format='Torch'):
        self.__graph = graph
        self.graph = self.__graph
        self.testgraphs = testgraphs
        self.symmetry = symmetry
        self.induced = induced
        self.undirected = undirected
        self.format = format

    def nx_graph(self):
        '''
        returns the target graph in networkX format
        '''
        if self.format == 'networkx':
            return self.__graph
        if self.format == 'Torch':
            return uts.to_networkx(self.__graph, to_undirected=self.undirected)
        else:
            raise Exception('Not implemented yet')

    def pyg_graph(self):
        '''
        returns the target graph in torch-geometric.data format
        '''
        if self.format == 'Torch':
            return self.__graph
        else:
            raise Exception("Not implemented yet")

    def draw_graph(self):
        '''
         returns a drawing of the encoded graph
        '''
        return nx.draw(self.nx_graph())

    def add(self, test_graph):
        '''
         Adds a graph to the set of test graphs
         parameters:
         test_graph (testGraph) : graph to be added
        '''
        self.testgraphs.add(test_graph)

    def add_from_iter(self, testgraphs):
        '''
         Adds graphs from an iterrable (e.g. list) to the set of test graphs
         parameters:
         testgraphs (testGraph) : iterator of graphs to be added
        '''
        new_testgraphs = set(testgraphs)
        self.testgraphs.update(new_testgraphs)

    def discard_testgraph(self, test_graph):
        self.testgraphs.discard(test_graph)

    def clear_all_testgraphs(self):
        self.testgraphs.clear()

    def subIso(self, testgraph):
        raise NotImplementedError

    def subIsodict(self):
        raise NotImplementedError

    def num_encoder(self, format='Torch'):
        '''
        Returns the subgraph isomorphism vector in R^|testgraphs| given by (|subgraphIso(F,G)|)_F
        '''
        raise NotImplementedError

    def pullback(self, testgraph, embedding,  format='Torch'):
        # returns the testgraph with features pulled back from the target graph along a map embedding
        '''
        input:
        testgraph (testGraph): a testgraph 
        embedding (gt.vertexpropertymap) :  an embedding of testgraph into the graph
        returns:
        subgraph (torch-geometric.data): testgraph as pyg data with features pulled along embedding
        '''
        raise NotImplementedError


class grandEmbedding(Embedding):
    '''
    A class for for handeling the embeddings from subgraph isomorphism. 
    '''

    def __init__(self, graph: Data, *kwargs):
        super().__init__(graph, *kwargs)

    def subIso(self, testgraph):
        '''
        params:
        testgraph (testgraph): a testgraph to map from
        returns: an itertable of monomorphisms from self to testgraph
        '''
        graph = super(grandEmbedding, self).nx_graph()
        return iter(find_motifs(testgraph.nx_graph(), graph, limit=testgraph.bound))

    def num_encoder(self, format='Torch'):
        '''
        Returns the subgraph isomorphism vector in R^|testgraphs| given by (|subgraphIso(F,G)|)_F
        '''
        graph = super(grandEmbedding, self).nx_graph()

        def num_all_subisos(x): return find_motifs(
            x.nx_graph(), graph, limit=x.bound, count_only=True)

        if format == 'Torch':
            return torch.tensor([num_all_subisos(test) for test in self.testgraphs])
        elif format == 'numpy':
            return np.array([num_all_subisos(test) for test in self.testgraphs])
        else:
            raise NotImplementedError("Format not supported")

    def __ghc_agg(self, test):
        dict_indices = map(lambda x: x.values(), self.subIso(test))
        indices = map(lambda x: list(x), dict_indices)

        tensorlist = [torch.prod(self.graph.x[idx], dim=0) for idx in indices]

        if len(tensorlist) == 0:
            num_node_features = self.graph.num_node_features
            return torch.zeros(num_node_features)

        test_agg = torch.stack(tensorlist)

        return torch.sum(test_agg, dim=0)

    def ghc_encoder(self, format='Torch'):
        '''
        An encoder in the style of the GHC paper
        returns: a concatanated tensor (or ndarray) multiplying coordinate functions of node featurs
        $ \sum_{f\in hom(F, self.graph)} \prod_{i\in V(F)} x_(f(i)) $
        for all F in testgraphset
        '''
        if self.graph.num_node_features == 0:
            return self.num_encoder(format=format)
        embedding_tensor = torch.stack(
            [self.__ghc_agg(test) for test in self.testgraphs])

        embedding_vector = embedding_tensor.flatten()
        if format == 'Torch':
            return embedding_vector
        elif format == 'numpy':
            return embedding_vector.detach().numpy()
        else:
            raise NotImplementedError("Format not supperted")

    def encoder(self, agg, format='Torch'):
        if self.graph.num_node_features == 0:
            return self.num_encoder(format=format)
        embedding_tensor = torch.stack(
            [self.__ghc_agg(test) for test in self.testgraphs])

        embedding_vector = embedding_tensor.flatten()
        if format == 'Torch':
            return embedding_vector
        elif format == 'numpy':
            return embedding_vector.detach().numpy()
        else:
            raise NotImplementedError("Format not supperted")


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    graph = uts.from_networkx(nx.cycle_graph(2))
    embd = grandEmbedding(graph)
