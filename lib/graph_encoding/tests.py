import unittest

from . import encoding
import networkx as nx

import torch
from torch_geometric.data import Data
import torch_geometric.utils as uts

import numpy as np

import itertools


class TestclassTestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.cycle_graph(3)
        self.testgraph = encoding.testGraph(self.graph)


class TestclassEmbedding(unittest.TestCase):
    def setUp(self):
        # skip testing the base class directly
        if self.__class__ is TestclassEmbedding:
            self.skipTest('Run no tests in base class')


class TestclassgrandEmbedding(TestclassEmbedding):
    def setUp(self):
        self.g = nx.complete_graph(5)
        self.graph = uts.from_networkx(self.g)
        self.graph.x = torch.tensor(
            [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0]])
        self.gembed = encoding.grandEmbedding(self.graph)

        self.sm_graph = uts.from_networkx(nx.complete_graph(3))
        self.sm_graph.x = torch.tensor(
            [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
        self.sm_graph.edge_attr = torch.tensor([[1], [2], [1], [3], [2], [3]])
        self.sm_gembed = encoding.grandEmbedding(self.sm_graph)

    def test_init(self):
        self.assertEqual(self.gembed.nx_graph().edges, self.g.edges)
        self.assertEqual(self.graph.num_node_features, 2)

    def test_add_single_vertex(self):
        self.gembed.clear_all_testgraphs()

        single_vertex = encoding.testGraph(
            nx.complete_graph(1), graph_name='single_vertex', limit=None)

        self.gembed.add_single_vertex()
        self.assertEqual(self.gembed.testgraphs, set({single_vertex}))

        self.gembed.clear_all_testgraphs()

    def test_add_cycles(self):
        self.gembed.clear_all_testgraphs()

        test_cycles = [encoding.testGraph(nx.cycle_graph(
            n), graph_name=f'c_{n}', limit=None) for n in range(3, 6)]

        self.gembed.add_cycles()
        self.assertEqual(self.gembed.testgraphs, set(test_cycles))

        self.gembed.clear_all_testgraphs()

    def test_add_cliques(self):
        self.gembed.clear_all_testgraphs()

        test_cliques = [encoding.testGraph(nx.complete_graph(
            n), graph_name=f'K_{n}', limit=None) for n in range(4, 5)]

        self.gembed.add_cliques()
        self.assertEqual(self.gembed.testgraphs, set(test_cliques))

        self.gembed.clear_all_testgraphs()

    def test_add_trees(self):
        self.gembed.clear_all_testgraphs()

        def make_tree(tree, n, m): return encoding.testGraph(
            tree, graph_name=f'tree of size {n}, number {m}', limit=None)

        def make_non_is_trees(n): return [make_tree(
            tree, n, m) for m, tree in enumerate(list(nx.nonisomorphic_trees(n)))]

        test_trees = list(itertools.chain.from_iterable(
            [make_non_is_trees(n) for n in range(2, 6)]))

        self.gembed.add_trees()
        self.assertEqual(self.gembed.testgraphs, set(test_trees))

        self.gembed.clear_all_testgraphs()

    def test_subIso(self):
        v = encoding.testGraph(nx.complete_graph(1), 'singe_vertex')
        node_maps = [{0: 0}, {0: 1}, {0: 2}, {0: 3}, {0: 4}]

        self.assertListEqual(list(self.gembed.subIso(v)), node_maps)

    def test_num_encoding(self):
        self.gembed.clear_all_testgraphs()
        self.gembed.add_single_vertex()
        self.assertEqual(np.array(5), self.gembed.num_encoder(format='numpy'))
        self.gembed.clear_all_testgraphs()

    def test_ghc_encoder(self):
        self.gembed.clear_all_testgraphs()
        self.gembed.add_single_vertex()
        expected_v = self.gembed.ghc_encoder(format='numpy')
        self.assertTrue(np.array_equal(
            np.array([5, 2]), expected_v), msg=f'{expected_v}')
        self.gembed.clear_all_testgraphs()
        self.gembed.add_trees(stop=3)
        expected_t = self.gembed.ghc_encoder(format='numpy')
        self.assertTrue(np.array_equal(
            np.array([20, 2]), expected_t), msg=f'{expected_t}')
        self.gembed.clear_all_testgraphs()

    def test_lagrangian_encoder(self):
        self.gembed.clear_all_testgraphs()
        self.gembed.add_single_vertex()
        expected_v = self.gembed.lagrangian_encoder(format='numpy')
        self.assertTrue(np.array_equal(
            np.array([5, 2]), expected_v), msg=f'{expected_v}')
        self.gembed.clear_all_testgraphs()
        self.gembed.add_trees(stop=3)
        expected_t = self.gembed.lagrangian_encoder(format='numpy')
        self.assertTrue(np.array_equal(
            np.array([20, 2]), expected_t), msg=f'{expected_t}')
        self.gembed.clear_all_testgraphs()

    def test_tensor_v_encoder(self):
        self.sm_gembed.clear_all_testgraphs()
        edge = encoding.testGraph(nx.complete_graph(2), graph_name='edge')
        self.sm_gembed.add(edge)
        embedding_list = self.sm_gembed.tensor_v_encoder()
        embedding_sizes = list(
            map(lambda x: list(x.size()), embedding_list))
        self.assertListEqual([[10]], embedding_sizes)
        self.sm_gembed.clear_all_testgraphs()

        self.sm_gembed.add_single_vertex()
        self.sm_gembed.add(edge)
        embedding_list = self.sm_gembed.tensor_v_encoder()
        embedding_sizes = list(
            map(lambda x: list(x.size()), embedding_list))
        #self.assertEqual(14, sum(embedding_sizes))
        self.sm_gembed.clear_all_testgraphs()

    def test_lagrangian_edge_encoder(self):
        self.sm_gembed.clear_all_testgraphs()
        edge = encoding.testGraph(nx.complete_graph(2), graph_name='edge')
        self.sm_gembed.add(edge)
        encoded_vector = self.sm_gembed.lagrangian_edge_encoder(format='numpy')
        self.assertEqual(5, encoded_vector.shape[0])


if __name__ == "__main__":
    unittest.main()
