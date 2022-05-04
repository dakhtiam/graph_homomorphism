#!/usr/bin/env python
# coding: utf-8

# # Graphs encoder with subgraph isomorphism numbers

# In[4]:


import torch

import numpy as np
import matplotlib.pyplot as plt
import itertools

import networkx as nx 
from networkx.algorithms.isomorphism import ISMAGS, GraphMatcher


# In[ ]:





# Encoding a graph into a vector in R^|F_set| given by (subgraphIso(F,G)) for F in Fset. 
# 
# Note that the standard ISMAGS has the order (graph,subgraph) in the arguments.  

# In[11]:


class SubGraph:

    def __init__(self, G, F_set):
        self.graph = G
        self.test_graphs = F_set

    def counts(self,symmetry = True):
        pass

    def shom(self,F):
        # get iterator over subgraph homomorphisms F->G.  
        pass


# In[ ]:





# In[15]:


def Simple_Encoder(G, F_set ,symmetry = True):
    '''
    This function returns the subgraph isomorphism vector in R^|F_set| given by (subgraphIso(F,G)) 
    
    Inputs:
        G: Graph
        F_set: The set of test subgraphs
        symmetry (default = True) : counting embeddings F->G up to automorphisms of F or not. 
    Output:
        Enc(G): (|F_set|, ) vector of subgraph isomorphism counting
    '''
    Isom_G = [ISMAGS(G,F) for F in F_set]
    numIso_G = map(lambda x: len(list(x.find_isomorphisms(symmetry))), Isom_G)

    return torch.tensor(numIso_G )

