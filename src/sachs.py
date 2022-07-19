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

DATA_PATH = "datasets/sachs/"
PATH = "sachs_experiments/"

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

node_names = ["RAF", "MEK", "ERK", "PLcg", "PIP2", "PIP3", "PKC", "AKT", "PKA", "JNK", "P38"]
var_names = ["praf", "pmek", "p44/42", "plcg", "PIP2", "PIP3", "PKC", "pakts473", "PKA", "pjnk", "P38"]

nodes_vars = dict(zip(node_names, var_names))

# --------------------------------------------------------------------
# DAG estimates from the literature

dag_consensus = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# cpdag_icp = np.array(
#     [
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
#     ]
# )

# dag_eaton = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

# dag_reconstructed = np.array(
#     [
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

# dag_mooij = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

# DAGs = []

# DAGs += [("consensus", dag_consensus), ("eaton", dag_eaton), ("reconstructed", dag_reconstructed), ("mooij", dag_mooij)]

# for i, dag in enumerate(utils.all_dags(cpdag_icp)):
#     DAGs.append(("dantzig-%d" % (i + 1), dag))


# --------------------------------------------------------------------
# Auxiliary functions


def process_data():
    """Process the data from the .csv files into a single np.array where
    the columns are in the same order as the DAGs."""
    dataframes = [pd.read_csv(DATA_PATH + f) for f in filenames]
    data = [df[var_names].to_numpy() for df in dataframes]
    np.savez(DATA_PATH + "sachs_data", *data)


def load_data(normalize=True):
    f = np.load(DATA_PATH + "sachs_data.npz")
    data = list(f.values())
    if normalize:
        pooled = np.vstack(data)
        mean = pooled.mean(axis=0)
        std = pooled.std(axis=0)
        data = [(X - mean) / std for X in data]
    return data


# --------------------------------------------------------------------
# Run experiments


if __name__ == "__main__":
    splits = list(range(50))
    # Run without pruning
    run_multisplit_experiments(prune_edges=True, splits=splits)
    # Run with pruning
    run_multisplit_experiments(prune_edges=False, splits=splits)
