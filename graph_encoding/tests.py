import unittest

import encoding
import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data
import torch_geometric.utils as uts


class TestclassTestGraph(unittest.TestCase):
    pass


class TestclassEmbedding(unittest.TestCase):
    def setUp(self):
        # skip testing the base class directly
        if self.__class__ is TestclassEmbedding:
            self.skipTest('Run no tests in base class')


class TestclassgrandEmbedding(TestclassEmbedding):
    def setUp(self):
        self.g = nx.cycle_graph(4)
        self.graph = uts.from_networkx(self.g)
        self.gembed = encoding.grandEmbedding(self.graph)

    def test_init(self):
        self.assertEqual(self.gembed.nx_graph().edges, self.g.edges)

    def test_add_cycles(self):
        self.gembed.add_cycles()
        test_cycles = [encoding.testGraph(nx.cycle_graph(
            n), graph_name=f'c_{n}', limit=None) for n in range(3, 6)]
        self.assertEqual(self.gembed.testgraphs, set(test_cycles))


if __name__ == "__main__":
    unittest.main()
