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
import time
import sachs

DATA_PATH = "sachs/taeb_2022/"

# --------------------------------------------------------------------
# Variable names

filenames = [
    "b2camp.csv",
    "cd3cd28+aktinhib.csv",
    "cd3cd28+g0076.csv",
    "cd3cd28+ly.csv",
    "cd3cd28+psitect.csv",
    "cd3cd28+u0126.csv",
    "cd3cd28.csv",
    "pma.csv",
]

NODE_NAMES = ["RAF", "MEK", "ERK", "PLcg", "PIP2", "PIP3", "PKC", "AKT", "PKA", "JNK", "P38"]
var_names = ["praf", "pmek", "p44/42", "plcg", "PIP2", "PIP3", "PKC", "pakts473", "PKA", "pjnk", "P38"]

assert sachs.node_names == NODE_NAMES

# --------------------------------------------------------------------
# Auxiliary functions


def _process_data():
    """Process the data from the .csv files into a single np.array where
    the columns are in the same order as the DAGs."""
    dataframes = [pd.read_csv(DATA_PATH + f) for f in filenames]
    data = [df[var_names].to_numpy() for df in dataframes]
    # Save data
    filename = DATA_PATH + "sachs_data_taeb_2022"
    np.savez(filename, *data)
    print("Saved dataset to %s.npz" % filename)


def load_data(normalize=False):
    return sachs.load_data(DATA_PATH + 'sachs_data_taeb_2022.npz', normalize)


def prepare_experiments_directory(path, graph_name, normalize=False):
    path += "" if path[-1] == "/" else "/"
    directory_name = "dataset_%d_sachs_taeb_2022_normalized:%s" % (time.time(), normalize)
    graph = sachs.DAGs[graph_name]
    data = load_data(normalize)
    sachs.prepare_experiments_directory(path + directory_name, data, graph)
