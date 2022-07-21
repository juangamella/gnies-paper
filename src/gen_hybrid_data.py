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

"""Module to generate "hybrid data" using a real dataset and
distributional random forests (https://arxiv.org/pdf/2005.14458.pdf).

"""
import argparse
import numpy as np
import time
import os
import src.utils as utils
import gc
from src.bayesian_networks import DRFNetwork, GaussianNetwork

# --------------------------------------------------------------------
# Auxiliary functions, BayesianNetwork class

_GRAPH_TYPE_ERROR = "graph must be a 2-dimensional numpy.ndarray"
_DATA_TYPE_ERROR = "data must be a list of 2-dimensional numpy.ndarray"
_N_TYPE_ERROR = "n must be a positive int or list of positive ints"
_BOOTSTRAP_TYPE_ERROR = "data must be a 1-dimension numpy.ndarray"
# TODO: Docstrings


def load_data(path, normalize=False):
    f = np.load(path)
    if isinstance(f, np.ndarray):  # supposedly .npy file
        data = list(f)
    else:  # supposedly .npz file
        data = list(f.values())
    if normalize:
        pooled = np.vstack(data)
        mean = pooled.mean(axis=0)
        std = pooled.std(axis=0)
        data = [(X - mean) / std for X in data]
    return data

# --------------------------------------------------------------------
# Parse input parameters


# Definitions and default settings
arguments = {
    # Execution parameters
    "cluster": {"default": False, "type": bool},
    "runs": {"default": 10, "type": int},
    "seed": {"default": 42, "type": int},
    "tag": {"type": str},
    "debug": {"default": False, "type": bool},
    "directory": {"default": "hybrid_experiments/", "type": str},
    # Sampling parameters
    "type": {"type": str, "default": "drf"},
    "standardize": {"type": bool, "default": False},
    "dataset": {"type": str},
    "graph": {"type": str},
    "n": {"default": "10", "type": str},
}

# Parse settings from input
parser = argparse.ArgumentParser(description="Run experiments")
for name, params in arguments.items():
    if params["type"] == bool:
        options = {"action": "store_true"}
    else:
        options = {"action": "store", "type": params["type"]}
    if "default" in params:
        options["default"] = params["default"]
    parser.add_argument("--" + name, dest=name, required=False, **options)

args = parser.parse_args()

# Parameters that will be excluded from the filename (see parameter_string function above)
excluded_keys = ["debug", "cluster", "dataset", "graph"]

print(args)  # For debugging

# --------------------------------------------------------------------
# Set up the directory to store generated data

args.directory += "" if args.directory[-1] == "/" else "/"
directory = args.directory + "dataset_%d%s/" % (time.time(),
                                                utils.parameter_string(args, excluded_keys))

# If running on the Euler cluster, store results in the scratch directory
if args.cluster:
    directory = "/cluster/scratch/gajuan/" + directory

os.makedirs(directory)

# Load graph and dataset, and store relevant information
graph = np.load(args.graph)
data = load_data(args.dataset, normalize=args.standardize)
Ns = sorted([int(n) for n in args.n.split(",")])
args.p = len(graph)
to_save = {"dataset": args.dataset, "n_cases": 1, "cases": [graph],
           "runs": args.runs, "Ns": Ns, "args": args, "graph": graph}

filename = directory + utils.INFO_FILENAME
utils.write_pickle(filename, to_save)
print('\nSaved test cases + info to "%s"' % filename)


# --------------------------------------------------
# Generate data for each run

if args.type == "drf":
    network = DRFNetwork(graph, data, verbose=True)
else:
    network = GaussianNetwork(graph, data, verbose=True)

print("---------------------------------")
print("Data\n")
print("Data has %d samples with sizes %s" % (network.e, network.Ns))


print("---------------------------------")
print("Graph\n")
for row in network.graph:
    print(" " * 5, (row != 0).astype(int))
print()

for n in Ns:
    start = time.time()
    print("---------------------------------")
    print("Generating datasets for n=%d" % n)
    print("    generating data for run:")
    print(" " * 5, end="")
    for r in range(args.runs):
        print(r, end=" ")
        data = network.sample(n, random_state=args.seed)
        path = directory + utils.test_case_filename(n, 0, r)
        utils.data_to_bin(data, path, debug=args.debug)
        del data
        gc.collect()
    print("  done in %0.2f seconds" % (time.time() - start))

print('Stored dataset in "%s"' % directory)
