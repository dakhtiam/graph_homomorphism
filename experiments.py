"""
Set up measurements for different pattern graphs
"""
# imports
from optparse import Option
import string
from tokenize import String
import numpy as np
# import torch
# from torch_geometric.data import Data
# from torch_geometric.transforms import BaseTransform
# from torch_geometric.datasets import TUDataset, ZINC
# from ogb.graphproppred import PygGraphPropPredDataset

import networkx as nx
from lib.graph_encoding.encoding import grandEmbedding

from typing import Any, Dict, List, Optional
import itertools
from functools import partial
from dataclasses import dataclass

# imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# from tqdm import tqdm

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


###### data classes for loading data and estimation ########

@dataclass
class ExperimentScore:
    scores: Any
    nums: Dict[str, int]
    clf_name: string
    cv_num: Optional[int]

    def plot_cv_scores(self):
        if self.cv_num == None:
            raise ValueError('No cross-validation number supplied')
        mean = self.scores.mean()
        std = self.scores.std()
        width = 0.35
        labels = [f'G{n}' for n in range(1, self.cv_num+1)]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(labels, self.scores, width)
        ax.set_ylabel('Scores')
        ax.set_title('Cross validation scores for ' + self.clf_name)
        plt.axhline(y=mean, c='black', linewidth=0.7,
                    label=f'Err = {mean:.2f}' + u"\u00B1" + f'{std:.2f}')
        ax.legend()

        plt.show()
        print(f'Validation error = {mean:.2f}' +
              u"\u00B1" + f'{std:.2f}')


@dataclass
class EncodingData:
    X: np.ndarray
    y: np.ndarray
    nums: Dict[str, int]

    def __clf(self, clf_name, cv_num=None, random_state=42):
        if clf_name == 'SVC':
            base_clf = SVC(C=1, random_state=random_state)
        elif clf_name == 'Random_forest':
            base_clf = RandomForestClassifier(random_state=random_state)
        else:
            raise NotImplementedError("Classifier type not supported")
        return make_pipeline(StandardScaler(), base_clf)

    def calculate_single_split_score(self, clf_name, scoring='accuracy',
                                     random_state=42, test_size=0.25):
        # test-train split:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

        # fit data
        clf = self.__clf(clf_name)
        clf.fit(X_train, y_train)

        # return  scores
        train_score = clf.score(X_train, y_train.ravel())
        test_score = clf.score(X_test, y_test.ravel())

        scores = {'train_score': train_score, 'test_score': test_score}

        return ExperimentScore(scores=scores, nums=self.nums, clf_name=clf_name)

    def calculate_cv_scores(self, clf_name, cv_num, scoring='accuracy'):
        clf = self.__clf(clf_name)
        scores = cross_val_score(
            clf, self.X, self.y.ravel(), cv=cv_num, scoring=scoring)

        return ExperimentScore(scores=scores, nums=self.nums, clf_name=clf_name, cv_num=cv_num)

######### experiment pipelines #######


class patternExperiment():
    def __init__(self, dataset, folder_name,
                 encoder_name, limit_vertex=None,
                 n_trees=2, limit_trees=10000,
                 n_cycles=3, limit_cycles=10000,
                 n_cliques=4, limit_cliques=100,
                 loading_mode=False, dataset_name=None):
        self.dataset = dataset
        self.encoded_dataset = None
        if loading_mode == True:
            self.dataset_name = dataset_name
        else:
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

        self.__datadict: Dict[int, EncodingData] = {}
        self.__scorsdict: Dict[int, ExperimentScore] = {}

        self.num_of_experiments = (self.n_cliques-4) * \
            (self.n_cycles-3)*(self.n_trees - 2)

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

    def load_data(self, get_value=False):
        for i in range(self.num_of_experiments):
            X = np.load(self.folder_name + '/' +
                        self.encoder_name+'/' + f'{i}_X.npy')
            y = np.load(self.folder_name+'/' +
                        self.encoder_name+'/' + f'{i}_y.npy')
            nums = np.load(self.folder_name + '/'+self.encoder_name +
                           '/' + f'{i}_nums.npy', allow_pickle=True).item()
            result = EncodingData(X, y, nums)
            self.__datadict[i] = result

        if get_value == True:
            return self.__datadict

    def __evaluate_scors(self, clf_name, scoring, cv_num=None, random_state=42):
        if cv_num == None:
            def get_score(x): return x.calculate_single_split_score(
                clf_name, scoring, random_state=random_state)
        else:
            def get_score(x): return x.calculate_cv_scores(
                clf_name, cv_num, scoring)

        for i, data in self.__datadict.items():
            score_data = get_score(data)
            self.__scorsdict[i] = score_data

    def __save_score_data(self, i, score_data: ExperimentScore):
        np.save(self.folder_name + '/'+self.encoder_name +
                '/' + f'{i}_{score_data.clf_name}_scores.npy', score_data.scores)

    def load_score_data(self, clf_name):
        score_dict = {}
        for i in range(self.num_of_experiments):
            score_dict[i] = np.load(self.folder_name + '/'+self.encoder_name +
                                    '/' + f'{i}_{clf_name}_scores.npy')
        return score_dict

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

    def evaluation(self, clf_name, scoring, cv_num=None, random_state=42):
        self.load_data()
        self.__evaluate_scors(
            clf_name, scoring, cv_num=cv_num, random_state=random_state)

        for i, data in self.__scorsdict.items():
            self.__save_score_data(i, data)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
