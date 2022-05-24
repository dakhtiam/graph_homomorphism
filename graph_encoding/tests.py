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
        self.g = nx.cycle_graph(2)
        self.graph = uts.from_networkx(self.g)
        self.gembed = encoding.grandEmbedding(self.graph)

    def test_init(self):
        self.assertEqual(self.gembed.nx_graph().graph, self.g.graph)


if __name__ == "__main__":
    unittest.main()
