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

import argparse
import sempler
import sempler.generators as gen
import numpy as np
import src.utils as utils
import gnies.utils
import gc  # Garbage collector
import time
import os
import pickle
import src.metrics as metrics

# --------------------------------------------------------------------
# Auxiliary functions


# --------------------------------------------------------------------
# Parse input parameters


# Definitions and default settings
arguments = {
    # Execution parameters
    'directory': {'type': str},
    'n_workers': {'default': 1, 'type': int},
    'debug': {'default': False, 'type': bool},
    'chunksize': {'type': int, 'default': 1},
    # Other flags parameters
    'methods': {'type': str},
}

# Parse settings from input
parser = argparse.ArgumentParser(description='Run experiments')
for name, params in arguments.items():
    if params['type'] == bool:
        options = {'action': 'store_true'}
    else:
        options = {'action': 'store', 'type': params['type']}
    if 'default' in params:
        options['default'] = params['default']
    parser.add_argument("--" + name,
                        dest=name,
                        required=False,
                        **options)

args = parser.parse_args()
print(args)  # For debugging


# --------------------------------------------------------------------
# Compute metrics
methods = args.methods.split(',')

# Extract dataset information
info = utils.read_pickle(args.directory + utils.INFO_FILENAME)
n_cases = info['n_cases']

# Build ground truths
print("---------------------------------------------")
print("Building ground truths\n")
ground_truth_classes = np.empty(n_cases, dtype=object)
ground_truth_skeletons = np.empty(n_cases, dtype=object)
ground_truth_Is = np.empty(n_cases, dtype=object)
ground_truth_dags = np.empty(n_cases, dtype=object)
for i, (scm, interventions) in enumerate(info['cases']):
    print("  case %d" % i)
    print("    interventions :", interventions)
    # Extract union of intervention targets
    union = set()
    # Note: interventions are [{intervention_type: {target: (mean, variance)}*}]
    for intervention in interventions:
        parameters = list(intervention.values())[0]
        union |= set(list(parameters.keys()))
    print("    union of targets :", union)
    ground_truth_Is[i] = union
    # Extract true DAG
    dag = (scm.W != 0).astype(int)
    ground_truth_dags[i] = dag
    # Compute true equivalence class
    true_class = gnies.utils.imec(dag, union)
    print("    %d DAGs in the true equiv. class" % len(true_class))
    ground_truth_classes[i] = true_class
    # Compute skeleton of equiv. graphs
    ground_truth_skeletons[i] = gnies.utils.skeleton(dag)
    print("    done.\n")

ground_truths = {'skeletons': ground_truth_skeletons,
                 'classes': ground_truth_classes,
                 'dags': ground_truth_dags,
                 'Is': ground_truth_Is}

computed_metrics = {}

print("---------------------------------------------")
print("Computing metrics")

for method in methods:
    # Read the method's result file
    args, results = utils.read_pickle(
        args.directory + utils.compiled_results_filename(method))
    print("\n  method = %s" % method)
    print("     which was run with settings")
    print("       ", args)
    # Load necessary estimates from the method
    estimates = results['estimates']  # I-CPDAGs
    I_estimates = results['I_estimates']  # Sets of intervention targets
    method_metrics = {}  # results dictionary
    # ---------------------
    # Compute t1/t2 class metrics
    print("    computing class metrics")
    funs = [metrics.type_1_structc, metrics.type_2_structc]
    class_metrics = utils.compute_metrics(
        estimates, ground_truth_classes, funs, gnies.utils.all_dags)
    method_metrics.update(class_metrics)
    # -------------------------
    # Compute skeleton recovery
    print("    computing skeleton metrics")
    funs = [metrics.type_1_skeleton, metrics.type_2_skeleton]
    skeleton_metrics = utils.compute_metrics(
        estimates, ground_truth_skeletons, funs, gnies.utils.skeleton)
    method_metrics.update(skeleton_metrics)
    # ------------------------------------
    # Compute intervention target recovery
    print("    computing intervention target metrics")
    funs = [metrics.type_1_I, metrics.type_2_I]
    I_metrics = utils.compute_metrics(
        I_estimates, ground_truth_Is, funs, lambda I: I)
    method_metrics.update(I_metrics)
    # -----------------------------
    # Compute elapsed times metrics
    print("    computing elapsed time")
    method_metrics['times'] = results['times']
    # Store results for this method
    computed_metrics[method] = method_metrics

print("Done.")

path = args.directory + 'metrics.pickle'  # % time.time()
print("Saved results to", path)
utils.write_pickle(path, (ground_truths, computed_metrics))
