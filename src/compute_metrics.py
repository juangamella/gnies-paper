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
import numpy as np
import src.utils as utils
import gnies.utils
import src.metrics as metrics

# --------------------------------------------------------------------
# Auxiliary functions


def all_dags(PDAG):
    """A wrapper for gnies.utils.all_dags but with protection against too
    large MECs"""
    try:
        return gnies.utils.all_dags(PDAG, max_combinations=int(1e4))
    except MemoryError as e:
        print(' '*8, e)
        return None
    except ValueError as e:
        print(' '*8, e)
        return None

def compute_metrics(estimates, ground_truth, metric_functions, trans_function, debug=False):
    """Given a set of estimates, a function `trans_function` to transform
    them if necessary, compute for every metric in `metric_functions`
    their score when compared to the ground truth"""
    assert len(ground_truth) == len(estimates)
    # Initialize the dictionary of arrays where results will be stored
    computed_metrics = dict((metric, np.empty_like(
        estimates, dtype=float)) for metric in metric_functions)
    # Iterate over each test case
    for i, case_estimates in enumerate(estimates):
        if debug:
            print(' '*5, "computing for case %d/%d" % (i, len(estimates)), end='\r')
        # Transform the estimates associated to this case
        trans_function = utils.if_none(trans_function)
        transformed_estimates = [trans_function(
            estimate) for estimate in case_estimates.flatten()]
        # Compute the requested metrics for each transformed estimate
        for metric in metric_functions:
            case_results = [utils.if_none(metric, np.nan)(estimate, ground_truth[i])
                            for estimate in transformed_estimates]
            # Store result in the corresponding array, reshaping the
            # flattened array of transformed estimates
            computed_metrics[metric][i] = np.reshape(
                case_results, computed_metrics[metric][i].shape)
    # Return
    print() if debug else None
    return computed_metrics

    
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
ground_truth_icpdags = np.empty(n_cases, dtype=object)
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
    # Compute the I-CPDAG
    icpdag = gnies.utils.dag_to_icpdag(dag, union)
    ground_truth_icpdags[i] = icpdag
    # Compute true equivalence class
    true_class = gnies.utils.all_dags(icpdag)
    print("    %d DAGs in the true equiv. class" % len(true_class))
    ground_truth_classes[i] = true_class
    # Compute skeleton of equiv. graphs
    ground_truth_skeletons[i] = gnies.utils.skeleton(dag)
    print("    done.\n")

ground_truths = {'skeletons': ground_truth_skeletons,
                 'classes': ground_truth_classes,
                 'dags': ground_truth_dags,
                 'Is': ground_truth_Is}

print("---------------------------------------------")
print("Computing metrics")

for method in methods:
    # Read the method's result file
    method_info, results = utils.read_pickle(
        args.directory + utils.compiled_results_filename(method))
    print("\n  method = %s" % method)
    print("     which was run with settings")
    print("       ", method_info)
    # Load necessary estimates from the method
    estimates = results['estimates']  # I-CPDAGs
    I_estimates = results['I_estimates']  # Sets of intervention targets
    method_metrics = {}  # results dictionary
    # ---------------------
    # Compute t1/t2 class metrics
    print("    computing class metrics")
    funs = [metrics.type_1_structc, metrics.type_2_structc]
    class_metrics = utils.compute_metrics(
        estimates, ground_truth_classes, funs, all_dags, debug=True)
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
    # ------------------------------------
    # Compute recovery of full I-MEC
    print("    computing recovery of full I-MEC")
    funs = [metrics.recovered_icpdag]
    recovery_metric = utils.compute_metrics(
        estimates, ground_truth_icpdags, funs, lambda x: x)
    method_metrics.update(recovery_metric)
    # -----------------------------
    # Compute proportion of times that method produced an estimate
    print("    computing method success")
    funs = [metrics.success_metric]
    success_metric = utils.compute_metrics(
        estimates, ground_truth_icpdags, funs, lambda x: x)
    method_metrics.update(success_metric)
    # -----------------------------
    # Compute elapsed times metrics
    print("    computing elapsed time")
    method_metrics['times'] = results['times']
    # Store results for this method
    path = args.directory + 'metrics_%s.pickle' % method
    print("  Saved results to", path)
    utils.write_pickle(path, (ground_truths, method_metrics))

print("Done.")
