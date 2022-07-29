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

"""
"""

import numpy as np
import pandas as pd
import sachs
import time

DATA_PATH = "sachs/wang_2017/"

filenames = [
    'iv=.txt',
    'iv=1.txt',
    'iv=3.txt',
    'iv=4.txt',
    'iv=6.txt',
    'iv=8.txt']

NODE_NAMES = ["RAF", "MEK", "ERK", "PLcg", "PIP2", "PIP3", "PKC", "AKT", "PKA", "JNK", "P38"]
var_names = ['Raf', 'Mek', 'Erk', 'PLCg', 'PIP2', 'PIP3', 'PKC', 'Akt', 'PKA', 'JNK', 'p38']

assert sachs.node_names == NODE_NAMES

idx_dict = dict((name,i) for i,name in enumerate(NODE_NAMES))

TARGETS = [None, "MEK", "PIP2", "PIP3", "AKT", "PKC"]
TARGETS_IDX =  [None if name is None else idx_dict[name] for name in TARGETS]

# --------------------------------------------------------------------
# Auxiliary functions

_expected_observations = [1755, 799, 810, 848, 911, 723]


def _process_data():
    """Process the data from the .csv files into a single np.array where
    the columns are in the same order as the DAGs."""
    dataframes = [pd.read_csv(DATA_PATH + f) for f in filenames]
    data = [df[var_names].to_numpy() for df in dataframes]
    # Test that data was correctly build
    assert [len(X) for X in data] == _expected_observations
    # Save data
    filename = DATA_PATH + "sachs_data_wang_2017"
    np.savez(filename, *data)
    print("Targets")
    for i, n in enumerate([len(X) for X in data]):
        print("  %s : %d observations" % (TARGETS[i], n))
    print("Saved dataset to %s.npz" % filename)


def load_data(normalize=True):
    return sachs.load_data(DATA_PATH + 'sachs_data_wang_2017.npz')


def prepare_experiments_directory(path, graph_name, normalize=False):
    path += "" if path[-1] == "/" else "/"
    directory_name = "dataset_%d_sachs_wang_2017_normalized:%s" % (time.time(), normalize)
    graph = sachs.DAGs[graph_name]
    data = load_data(normalize)
    sachs.prepare_experiments_directory(path + directory_name, data, graph)
