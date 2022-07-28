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

filenames = ['iv=.txt', 'iv=1.txt', 'iv=3.txt', 'iv=4.txt', 'iv=6.txt', 'iv=8.txt']
filenames = ['iv=.txt', 'iv=1.txt', 'iv=3.txt', 'iv=4.txt', 'iv=6.txt', 'iv=8.txt']

node_names = ["RAF", "MEK", "ERK", "PLcg", "PIP2", "PIP3", "PKC", "AKT", "PKA", "JNK", "P38"]
var_names = ['Raf', 'Mek', 'Erk', 'PLCg', 'PIP2', 'PIP3', 'PKC', 'Akt', 'PKA', 'JNK', 'p38']

idx_dict = dict((name,i) for i,name in enumerate(node_names))
target_names = [None, "MEK", "PIP2", "PIP3", "AKT", "PKC"]
expected_observations = [1755, 799, 810, 848, 911, 723]
target_indices = [None if name is None else idx_dict[name] for name in target_names]

print("Target indices: %s" % target_indices)

dataframes = [pd.read_csv(f) for f in filenames]
for df in dataframes:
    print(df.columns)
data = [df[var_names].to_numpy() for df in dataframes]
np.savez("sachs_data_wang_2017", *data)

print("Observations:", [len(X) for X in data])
print("Should match:", expected_observations)
print(" ", [len(X) for X in data] == expected_observations)

print('Compiled data into "sachs_data_wang_2017"')

# print('Preparing directory for evaluation - ground truth is consensus graph')

# os.makedirs('real_dataset_wang_2017_dag:consensus')
#     data = load_data(normalize)
#     n = 800  # Just used for the filenames
#     # Write data
#     filename = path + utils.test_case_filename(n, 0, 0)
#     utils.data_to_bin(data, path + utils.test_case_filename(n, 0, 0), debug=True)
#     # Write test case info
#     graph = dict(DAGs)[dag_name]
#     args = argparse.Namespace()
#     args.p = len(graph)
#     to_save = {"n_cases": 1, "cases": [graph], "runs": 1, "Ns": [n], "args": args,
#                "graph": graph}
#     filename = path + utils.INFO_FILENAME
#     utils.write_pickle(filename, to_save)
#     print('  saved test case info to "%s"' % filename)
#     # Write test_case_graph
#     filename = path + "consensus_graph"
#     utils.data_to_bin(graph, filename, debug=False)
#     print('  saved test graph to "%s"' % filename)
