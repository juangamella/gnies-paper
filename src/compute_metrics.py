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

use_metrics = [metrics.type_1_structc, metrics.type_2_structc]
methods = args.methods.split(',')

# Extract dataset information
info = utils.read_pickle(args.directory + utils.INFO_FILENAME)
n_cases, runs, Ns = info['n_cases'], info['runs'], info['Ns']
Ns_idx = dict(zip(sorted(Ns), range(len(Ns))))

# Build ground truths
print("---------------------------------------------")
print("Building ground truths\n")
size = (len(Ns), n_cases, runs)
ground_truth = np.empty(n_cases, dtype=object)
for i, (scm, interventions) in enumerate(info['cases']):
    print("  case %d" % i)
    print("    interventions :", interventions)
    union = set()
    for intervention in interventions:
        parameters = list(intervention.values())[0]
        union |= set(list(parameters.keys()))
    print("    union of targets :", union)
    true_class = gnies.utils.imec(scm.W, union)
    print("    %d DAGs in the true equiv. class" % len(true_class))
    ground_truth[i] = true_class
    print("    done.\n")

# assert not np.any(ground_truth == None)

# Build arrays to store metrics
shape = (len(Ns), n_cases, runs)
metrics = {}
for metric in use_metrics:
    metrics[metric] = {}


print("---------------------------------------------")
for method in methods:
    args, results = utils.read_pickle(
        args.directory + utils.compiled_results_filename(method))
    print("Computing metrics for method = %s" % method)
    print("   which was run with settings")
    print("     ", args)
    # Load estimates from the method
    estimates = results['estimates']
    # Compute the metrics for each test case
    computed_metrics = dict((metric, np.empty_like(
        estimates, dtype=float)) for metric in use_metrics)
    for i, case_estimates in enumerate(estimates):
        estimated_classes = [gnies.utils.all_dags(
            estimate) for estimate in case_estimates.flatten()]
        for metric in use_metrics:
            case_result = [metric(estimated_class, ground_truth[i])
                           for estimated_class in estimated_classes]
            computed_metrics[metric][i] = np.reshape(
                case_result, computed_metrics[metric][i].shape)
    print(computed_metrics)

# print(metrics)
