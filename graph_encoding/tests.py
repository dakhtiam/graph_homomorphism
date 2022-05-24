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
            self.skipTest('run No Tests In Base Class')


class TestclassgrandEmbedding(TestclassEmbedding):
    pass


if __name__ == "__main__":
    unittest.main()
