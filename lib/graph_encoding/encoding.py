"""
A mini-library to process graph encoding based on subgraph strcuture
"""
# imports
import string
import torch
from torch_geometric.data import Data
import torch_geometric.utils as uts

import numpy as np
import networkx as nx
from grandiso import find_motifs

from typing import Set
import itertools
from functools import partial
import scipy.special

####### testgraph and testset classes ##########


class testGraph:
    '''
    A class object for a test graph with descriptive features
    '''

    def __init__(self, graph, graph_name: string, limit=None, format='networkx'):
        '''
         Parameters:
         graph: a graph in the format "format" (default is networkX)
         graph_name (String): the name of a graph
         max_n (Int): a limit for the number of subgraphs taken
        '''
        self.name = graph_name
        self.__graph = graph
        self.limit = limit
        self.format = format

    def __eq__(self, other) -> bool:
        '''
        overriding dafault __eq__ because networkx hashes the same graph differently
        '''
        if self.name != other.name:
            return False
        if set(self.nx_graph().edges) != set(other.nx_graph().edges):
            return False
        if self.limit != other.limit:
            return False
        return True

    def __hash__(self) -> int:
        '''
        Implementing __hash__ since __eq__ is overriden
        '''
        return hash(self.name)  # hashing by name
        # return hash((self.name, self.limit))  # hashing by name and size
        # return hash((self.name, frozenset(self.nx_graph().edges), self.limit))

    def __str__(self) -> string:
        return self.name

    def nx_graph(self):
        '''
        returns the graph in networkX format
        '''
        return self.__graph

    def pyg_graph(self):
        if self.format != 'networkx':
            raise Exception('Not implemented yet')
        return uts.convert.from_networkx(self.__graph)

    def num_nodes(self):
        return self.nx_graph().number_of_nodes()

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
                 testgraphs: testGraphSet = set(),
                 symmetry=False,
                 induced=False,
                 undirected='True',
                 format='Torch'):
        self.__graph = graph
        self.graph = self.__graph
        if testgraphs == set():
            self.testgraphs = set()
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

    def add_from_iter(self, new_testgraphs):
        '''
         Adds graphs from an iterrable (e.g. list) to the set of test graphs
         parameters:
         testgraphs (testGraph) : iterator of graphs to be added
        '''
        self.testgraphs.update(new_testgraphs)

    def discard_testgraph(self, test_graph):
        self.testgraphs.discard(test_graph)

    def clear_all_testgraphs(self):
        self.testgraphs.clear()

    def add_single_vertex(self, limit=None):
        single_vertex = testGraph(
            nx.complete_graph(1), graph_name='single_vertex', limit=limit)

        self.add(single_vertex)

    def add_trees(self, start=2, stop=6, limit=None):
        '''
        Adding a set of all non-isomorphic trees of sizes [start, stop] to the testgraph set
        '''
        def make_tree(tree, n, m): return testGraph(
            tree, graph_name=f'tree of size {n}, number {m}', limit=limit)
        def make_non_is_trees(n): return [make_tree(
            tree, n, m) for m, tree in enumerate(list(nx.nonisomorphic_trees(n)))]
        test_trees = list(itertools.chain.from_iterable(
            [make_non_is_trees(n) for n in range(start, stop)]))

        self.add_from_iter(test_trees)

    def add_cycles(self, start=3, stop=6, limit=None):
        '''
        Adding a set cycles of length [start, stop] to the testgraph set
        '''
        test_cycles = [testGraph(nx.cycle_graph(
            n), graph_name=f'c_{n}', limit=limit) for n in range(start, stop)]
        self.add_from_iter(test_cycles)

    def add_cliques(self, start=4, stop=5, limit=None):
        test_cliques = [testGraph(nx.complete_graph(
            n), graph_name=f'K_{n}', limit=limit) for n in range(start, stop)]
        self.add_from_iter(test_cliques)

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
        return iter(find_motifs(testgraph.nx_graph(), graph, limit=testgraph.limit))

    def num_encoder(self, format='Torch'):
        '''
        Returns the subgraph isomorphism vector in R^|testgraphs| given by (|subgraphIso(F,G)|)_F
        '''
        graph = super(grandEmbedding, self).nx_graph()

        def num_all_subisos(x): return find_motifs(
            x.nx_graph(), graph, limit=x.limit, count_only=True)

        if format == 'Torch':
            return torch.tensor([num_all_subisos(test) for test in self.testgraphs])
        elif format == 'numpy':
            return np.array([num_all_subisos(test) for test in self.testgraphs])
        else:
            raise NotImplementedError("Format not supported")

    def __encoder(self, agg, format='Torch', flatten=True):
        if self.graph.num_node_features == 0:
            return self.num_encoder(format=format)

        embedding_list = [agg(test) for test in self.testgraphs]

        if len(embedding_list) == 0:
            raise ValueError('Expecting a non-empty list of testgraphs')
        if flatten == False:
            if format != 'Torch':
                raise NotImplementedError("Format not supperted")
            return embedding_list

        embedding_tensor = torch.stack(embedding_list, dim=0)
        embedding_tensor = embedding_tensor.flatten()
        if format == 'Torch':
            return embedding_tensor.contiguous()
        elif format == 'numpy':
            return embedding_tensor.detach().numpy()
        else:
            raise NotImplementedError("Format not supperted")

    def __ghc_agg(self, test):
        '''
        The local aggregation function for ghc_encoder
        '''
        dict_indices = map(lambda x: x.values(), self.subIso(test))
        indices = map(lambda x: list(x), dict_indices)

        tensorlist = [torch.prod(self.graph.x[idx], dim=0) for idx in indices]

        if len(tensorlist) == 0:
            num_node_features = self.graph.num_node_features
            return torch.zeros(num_node_features)

        test_agg = torch.stack(tensorlist, dim=0)

        return torch.sum(test_agg, dim=0)

    def ghc_encoder(self, format='Torch'):
        '''
        An encoder in the style of the GHC paper
        returns: a concatanated tensor (or ndarray) multiplying coordinate functions of node featurs
        $ \sum_{f\in hom(F, self.graph)} \oplus_{k=1..d}\prod_{i\in V(F)} x^k_(f(i)) $
        for F in testgraphset and d the dimension of node features
        '''
        return self.__encoder(self.__ghc_agg, format=format)

    def __lagrangian_agg(self, test: testGraph):
        '''
        Local aggregation for Lagrangian encoder
        '''
        if test.name == 'single_vertex':
            return self.__ghc_agg(test)
        dict_indices = self.subIso(test)
        test_edges = test.pyg_graph().edge_index.t()

        test_edge_list = [test_edges.clone().apply_(
            lambda x: dict[x]) for dict in dict_indices]
        if len(test_edge_list) == 0:
            num_node_features = self.graph.num_node_features
            return torch.zeros(num_node_features)

        test_edge_tensor = torch.stack(test_edge_list)
        x_edge_pair = self.graph.x[test_edge_tensor]

        x_edge_product = torch.prod(x_edge_pair, dim=-2)
        test_agg = torch.sum(x_edge_product, dim=-2)

        return torch.div(torch.sum(test_agg, dim=0), 2)

    def lagrangian_encoder(self, format='Torch'):
        '''
        An encdoer with local augmentation given by lagrangian functions:
        $ \sum_{f\in hom(F, self.graph)} \oplus_{k = 1..d}\sum_{(i,j)\in E(F)} x^k_(f(i))x^k_(f(j)) $
        for F in testgraphset and d the dimension of node features
        '''
        return self.__encoder(self.__lagrangian_agg, format=format)

    def __lagrangian_edge_agg(self, test):
        '''
        Local aggregation for Lagrangian encoder
        '''
        if test.name == 'single_vertex':
            return self.__ghc_agg(test)

        def to_str(x): return f'{x}'
        edge_strings = map(to_str, self.graph.edge_index.t().tolist())
        edge_dict = dict(map(lambda x: reversed(x), enumerate(edge_strings)))
        if test.name == 'single_vertex':
            return self.__ghc_agg(test)
        dict_indices = self.subIso(test)
        test_edges = test.pyg_graph().edge_index.t()

        test_edge_list = [test_edges.clone().apply_(
            lambda x: dict[x]) for dict in dict_indices]
        if len(test_edge_list) == 0:
            num_node_features = self.graph.num_node_features
            return torch.zeros(num_node_features)

        test_edge_tensor = torch.stack(test_edge_list)
        x_edge_pair = self.graph.x[test_edge_tensor]

        x_edge_product = torch.prod(x_edge_pair, dim=-2)
        test_agg = torch.sum(x_edge_product, dim=-2)
        node_contribution = torch.div(torch.sum(test_agg, dim=0), 2)

        test_edge_list = list(map(lambda x: x.tolist(), test_edge_list))

        test_edge_index = [list(map(lambda x: edge_dict[to_str(x)], edge_list))
                           for edge_list in test_edge_list]

        edge_features = torch.stack(
            [self.graph.edge_attr[edge] for edge in test_edge_index])

        edge_contribution = torch.unsqueeze(torch.sum(edge_features), 0)

        #print('edge_contribution = ', test_agg.size())
        return torch.cat((node_contribution, edge_contribution), dim=-1)

    def lagrangian_edge_encoder(self, format='Torch'):
        '''
        An encdoer with local augmentation given by lagrangian functions:
        $ \sum_{f\in hom(F, self.graph)} \oplus_{k = 1..d}\sum_{(i,j)\in E(F)} x^k_(f(i))x^k_(f(j)) $
        for F in testgraphset and d the dimension of node features
        '''
        return self.__encoder(self.__lagrangian_edge_agg, format=format)

    def __tensor_v_agg(self, test: testGraph):
        '''
        Calculats the index tensor for the tensor vertex encoding
        '''
        dict_indices = list(map(lambda x: x.values(), self.subIso(test)))
        d = self.graph.num_node_features
        if len(dict_indices) == 0:
            return torch.zeros(d)
        indices = map(lambda x: list(x), dict_indices)

        test_num_nodes = test.num_nodes()
        d_inx = torch.arange(d)
        indx_tensor = torch.combinations(
            d_inx, r=test_num_nodes, with_replacement=True).t().tolist()
        array_list = (list(zip(e, indx_tensor)) for e in indices)

        # cast self.graph.x
        def embedding_pretensor(array): return torch.stack(
            [self.graph.x[i] for i in array])

        pre_tensorlist = [embedding_pretensor(
            array) for array in array_list]

        # if len(pre_tensorlist) == 0:
        #   num_node_features = self.graph.num_node_features
        #  return torch.zeros(num_node_features)

        test_agg = torch.prod(torch.stack(pre_tensorlist, dim=0), dim=1)

        # return torch.zeros(d)
        return torch.sum(test_agg, dim=0).contiguous()

    def tensor_v_encoder(self, format='Torch'):
        '''
        An generalized encoder for incorporating node and subgraph features.
        returns: a concatanated tensor (or ndarray) multiplying coordinate functions of node featurs
        $ \sum_{f\in hom(F, self.graph)} Sym(1 \oplus \otimes_{i\in V(F)} x^k_(f(i))) $
        for F in testgraphset and d the dimension of node features

        The formula inspired by the definition of Lovasz
        '''
        return self.__encoder(self.__tensor_v_agg, format=format, flatten=False)

    def tensor_v_encoding_fast(self, format='Torch'):
        pass
