# Copyright 2020 Juan Luis Gamella Martin

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import drf
import gnies.utils
import copy
import pandas as pd
import time
from gnies.scores.experimental import FixedInterventionalScore

_GRAPH_TYPE_ERROR = "graph must be a 2-dimensional numpy.ndarray"
_DATA_TYPE_ERROR = "data must be a list of 2-dimensional numpy.ndarray"
_N_TYPE_ERROR = "n must be a positive int or list of positive ints"
_BOOTSTRAP_TYPE_ERROR = "data must be a 1-dimension numpy.ndarray"

# --------------------------------------------------------------------
# Auxiliary functions


def bootstrap(n, data, random_state=None):
    """
    """
    # Check inputs
    if not isinstance(data, np.ndarray):
        raise TypeError(_BOOTSTRAP_TYPE_ERROR)
    elif data.ndim != 1:
        raise ValueError(_BOOTSTRAP_TYPE_ERROR)

    # Sample with replacement
    rng = np.random.default_rng(random_state)
    sample = rng.choice(data, n, replace=True)
    return sample

# --------------------------------------------------------------------
# BayesianNetwork class


class BayesianNetwork:
    def __init__(self, graph, data, verbose=False):
        # Check inputs: graph
        if not isinstance(graph, np.ndarray):
            raise TypeError(_GRAPH_TYPE_ERROR)
        elif graph.ndim != 2:
            raise ValueError(_GRAPH_TYPE_ERROR)
        elif not gnies.utils.is_dag(graph):
            raise ValueError("graph is not a DAG.")

        # Check inputs: data
        if not isinstance(data, list):
            raise TypeError(_DATA_TYPE_ERROR)
        else:
            for sample in data:
                # Check every sample is a numpy.ndarray
                if not isinstance(sample, np.ndarray):
                    raise TypeError(_DATA_TYPE_ERROR)
                # with two dimensions
                elif sample.ndim != 2:
                    raise ValueError(_DATA_TYPE_ERROR)
                # and the number of variables matches that in the graph
                elif sample.shape[1] != graph.shape[1]:
                    raise ValueError("graph and data have different number of variables")

        # Set parameters
        self.graph = (graph != 0).astype(int)
        self._data = copy.deepcopy(data)
        self.p = graph.shape[1]
        self.e = len(self._data)
        self.Ns = [len(sample) for sample in self._data]
        self._ordering = gnies.utils.topological_ordering(self.graph)

    def sample(self, n):
        # Check input: n
        if type(n) not in [int, list]:
            raise TypeError(_N_TYPE_ERROR)
        elif type(n) == int and n <= 0:
            raise ValueError(_N_TYPE_ERROR)
        elif type(n) == list:
            for i in n:
                if type(i) != int:
                    raise TypeError(_N_TYPE_ERROR)
                elif i <= 0:
                    raise ValueError(_N_TYPE_ERROR)
        return None


class DRFNetwork(BayesianNetwork):
    def __init__(self, graph, data, verbose=False):
        super().__init__(graph, data, verbose)

        # Fit distributional random forests, i.e. one per (node with
        # parents) x environment
        if verbose:
            start = time.time()
            print("-------------------------------------")
            print("Fitting distributional random forests")
        self._random_forests = np.empty((self.p, self.e), dtype=object)
        for i in range(self.p):
            parents = gnies.utils.pa(i, self.graph)
            print("  node %d/%d - parents %s     " % (i + 1, self.p, parents)) if verbose else None
            # Don't fit a DRF if node is a source node
            if parents != set():
                for k in range(self.e):
                    print("    fitting environment %d/%d" %
                          (k + 1, self.e), end="  \r") if verbose else None
                    # Using default values from DRF repository
                    DRF = drf.drf(min_node_size=15, num_trees=2000, splitting_rule="FourierMMD")
                    Y = pd.DataFrame(self._data[k][:, i])
                    X = pd.DataFrame(self._data[k][:, sorted(parents)])
                    DRF.fit(X, Y)
                    # print(DRF.info())
                    self._random_forests[i, k] = DRF
        print("Done in %0.2f seconds           " % (time.time() - start)) if verbose else None

    def sample(self, n, random_state=None):
        super().sample(n)  # Checks inputs
        n = [n] * self.e if type(n) == int else n
        # Generate a sample for each environment
        sampled_data = []
        for k in range(self.e):
            sample = np.zeros((n[k], self.p), dtype=float)
            for i in self._ordering:
                if self._random_forests[i, k] is None:
                    # Node has no parents, generate a sample using bootstrapping
                    sample[:, i] = bootstrap(n[k], self._data[k][:, i], random_state=random_state)
                else:
                    parents = gnies.utils.pa(i, self.graph)
                    new_data = pd.DataFrame(sample[:, sorted(parents)])
                    forest = self._random_forests[i, k]
                    output = forest.predict(n=1, functional="sample", newdata=new_data)
                    sample[:, i] = output.sample[:, 0, 0]
            sampled_data.append(sample)
        return sampled_data


class GaussianNetwork(BayesianNetwork):
    def __init__(self, graph, data, verbose=False):
        super().__init__(graph, data, verbose)

        # Fit distributional random forests, i.e. one per (node with
        # parents) x environment
        if verbose:
            start = time.time()
            print("-------------------------------------")
            print("Fitting Gaussian model")
        full_I = set(range(self.p))
        score_class = FixedInterventionalScore(self._data, full_I, centered=False)
        self.B, self.noise_term_means, self.noise_term_variances = score_class._mle_full(self.graph, [
                                                                                         full_I] * self.e)
        print("Done in %0.2f seconds" % (time.time() - start)) if verbose else None

    def sample(self, n, random_state=None):
        super().sample(n)  # Checks inputs
        n = [n] * self.e if type(n) == int else n
        # Generate a sample for each environment
        sampled_data = []
        rng = np.random.default_rng(random_state)
        for k in range(self.e):
            sample = np.zeros((n[k], self.p), dtype=float)
            for i in self._ordering:
                mean = self.noise_term_means[k, i]
                std = np.sqrt(self.noise_term_variances[k, i])
                noise_sample = rng.normal(mean, std, n[k])
                sample[:, i] = sample @ self.B[:, i] + noise_sample
            sampled_data.append(sample)
        return sampled_data
