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


class testGraphSet:
    def __init__(self, n_cycles=3, n_trees=6, n_cliques=2, *kwargs):
        self.n_cycles = n_cycles
        self.n_trees = n_trees
        self.n_cycles = n_cliques

    def cycles(self):
        pass

    def list(self):
        '''
        Returns the set of testgraphs as a list of testGraph objects
        '''
        pass

    def gen(self):
        '''
        Returns a generator of testGraph objects over the test graphs
        '''
        pass

####### generic Embedding classes ##########


class Embedding():
    '''
    A parent class for for handeling the embeddings from subgraph isomorphism. 
    A specific class using aparticular graph homomorphism computation algorithm or library 
    inherits this class
    '''

    def __init__(self, graph,
                 testgraphs=None,
                 symmetry=True,
                 induced=False,
                 undirected='True',
                 format='Torch'):
        self.__graph = graph
        self.graph = self.__graph
        if testgraphs == None:
            self.testgraphs = {}
        else:
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
        self.testgraphs[test_graph.name] = test_graph

    def add_from_iter(self, testgraphs):
        '''
         Adds graphs from an iterrable (e.g. list) to the set of test graphs
         parameters:
         testgraphs (testGraph) : iterator of graphs to be added
        '''
        dict_new_testgraphs = {F.name: F for F in testgraphs}
        self.testgraphs.update(dict_new_testgraphs)

    def subIso(self, testgraph):
        raise NotImplementedError

    def subIsodict(self):
        raise NotImplementedError

    def num_encoder(self, format='Torch'):
        '''
        Returns the subgraph isomorphism vector in R^|testgraphs| given by (|subgraphIso(F,G)|)_F
        '''
        raise NotImplementedError

    def __pullback(self, testgraph, embedding,  format='Torch'):
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

    def __init__(self, *kwargs):
        super().__init__(*kwargs)

    def subIso(self, testgraph):
        '''
        params:
        testgraph (testgraph): a testgraph to map from
        returns:
        returns a list of monomorphisms of the form
        "testgraph_id": "graph_id"
        '''
        graph = super(grandEmbedding, self).nx_graph()
        return iter(find_motifs(testgraph.nx_graph(), graph, limit=testgraph.bound))
    # def subIso(self, testgraph):
        # a public version of subIso, might be revoked in the future
        # return self.subIso(testgraph)

    def num_encoder(self, format='Torch'):
        '''
        Returns the subgraph isomorphism vector in R^|testgraphs| given by (|subgraphIso(F,G)|)_F
        '''
        graph = super(grandEmbedding, self).nx_graph()

        def num_auto(x): return find_motifs(
            x.nx_graph(), x.nx_graph(), limit=x.bound, count_only=True)
        def num_all_subisos(x): return find_motifs(
            x.nx_graph(), graph, limit=x.bound, count_only=True)

        if format == 'Torch':
            return torch.tensor([num_all_subisos(test) for test in self.testgraphs.values()])
        elif format == 'numpy':
            return np.array([num_all_subisos(test) for test in self.testgraphs.values()])
        else:
            raise Exception("Format not supported")

    def ghc_agg(self, test):
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
        '''
        if self.graph.num_node_features == 0:
            return self.num_encoder(format=format)
        embedding_tensor = torch.stack(
            [self.ghc_agg(test) for test in self.testgraphs.values()])

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
