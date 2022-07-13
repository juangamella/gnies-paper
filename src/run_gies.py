# Copyright 2022 Juan L. Gamella

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

"""Script to run GIES on the interventional data, using the
ground-truth targets.

"""

# TODO

import numpy as np
import time
from datetime import datetime
import argparse
import os
import multiprocessing
import gnies.utils
import ges
import src.utils as utils
import traceback

METHOD_NAME = 'gies'

# --------------------------------------------------------------------
# Auxiliary functions
if __name__ != '__main__':
    msg = "Not running as a script, i.e. python -m package.module"
    raise Exception(msg)

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
    'compile_only': {'type': bool, 'default': False},
    # GIES parameters
    'one_run': {'default': False, 'type': bool},
    'phases': {'default': 'fbt', 'type': str},
    'lambda_lo': {'default': 0.5, 'type': float},
    'lambda_hi': {'default': 0.5, 'type': float},
    'n_lambdas': {'default': 1, 'type': int},
    'gies_verbose': {'default': False, 'type': bool},
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

excluded_keys = [
    'store_history',
    'gies_verbose',
    'directory',
    'chunksize',
    'n_workers',
    'compile_only'
]
excluded_keys += [] if args.one_run else ['one_run']

# --------------------------------------------------------------------
# Run algorithm on samples

print("Visible workers: %d vs. args.n_workers %d" %
      (os.cpu_count(), args.n_workers))

# Extract dataset information
info = utils.read_pickle(args.directory + utils.INFO_FILENAME)
n_cases, runs, Ns, p = info['n_cases'], info['runs'], info['Ns'], info['args'].p

# Extract ground truth intervention targets
all_targets = []
# Note: interventions are [{intervention_type: {target: (mean, variance)}*}]ñ
for (_, interventions) in info['cases']:
    print(interventions)
    case_targets = [list(list(i.values())[0].keys()) for i in interventions]
    all_targets.append(case_targets)
    print(case_targets)

n_samples = n_cases * runs * len(Ns)
print("Dataset contains a total of %d samples from %d cases at sample sizes %s for %d runs" %
      (n_samples, n_cases, Ns, runs))

# --------------------------------------------------------------------

# ---------------------------------------------------------------------
# Build iterable, result_matrix_shape, iterable entry to index function

lmbdas = utils.hyperparameter_range(args.lambda_lo,
                                    args.lambda_hi,
                                    args.n_lambdas)
fields = [range(n_cases), lmbdas, Ns, range(runs)]

iterable = []
for (graph, lmbda, sample_size, run) in gnies.utils.cartesian(fields, dtype=object):
    iterable.append({'l': lmbda,
                     'n': sample_size,
                     'g': graph,
                     'r': run})
assert len(iterable) == n_samples * len(lmbdas)

# NOTE!!!: The case must be the first index (to easily recover ground
# truth when computing metrics)

result_matrix_shape = tuple(len(field) for field in fields)

Ns_idx = dict(zip(sorted(Ns), range(len(Ns))))
lambdas_idx = dict(zip(sorted(lmbdas), range(len(lmbdas))))


def case_info_to_indices(info):
    lmbda = info['l']
    n = info['n']
    return (info['g'], lambdas_idx[lmbda], Ns_idx[n], info['r'])


# ------------------
# Code to run method
gies_phases = []
for symbol in args.phases:
    if symbol == 'f':
        gies_phases.append('forward')
    elif symbol == 'b':
        gies_phases.append('backward')
    elif symbol == 't':
        gies_phases.append('turning')
    else:
        raise ValueError('Invalid symbol "%s" in args.gies_phases ("%s").' %
                         (symbol, args.gies_phases))

gies_options = {'iterate': not args.one_run,
                'phases': gies_phases,
                'debug': args.gies_verbose}

print("Running GIES with settings:")
print(" ", gies_options)
print("  on lambdas :", lmbdas)


def run_method(info, debug=False):
    """Takes an iterable entry and runs the algorithm accordingly"""
    # Load data
    n, graph, run = info['n'], info['g'], info['r']
    data_path = args.directory + utils.test_case_filename(n, graph, run)
    data = utils.load_bin(data_path)
    # Compute penalization parameter
    N = sum([len(X) for X in data])
    lmbda = info['l'] * np.log(N)
    # Retrieve intervention targets for GIES
    targets = all_targets[graph]
    # Run method
    start = time.time()
    score_class = gies.scores.GaussObsL0Pen(data, targets, lmbda=lmbda)
    output = gies.fit(score_class, **gies_options)
    elapsed = time.time() - start
    print("  Ran GIES on test case %s in %0.2f seconds." %
          (utils.serialize_dict(info), elapsed)) if debug else None
    # Store results
    estimated_icpdag, score = output
    result = {'estimate': estimated_icpdag,
              'estimated_I': [set(t) for t in targets],
              'lambda': lmbda,
              'n': n,
              'elapsed': elapsed}
    return result

# ---------------
# Process results


def process_results():
    print("Processing results:\n")
    # Common to all methods
    estimates = np.empty(result_matrix_shape, dtype=object)
    I_estimates = np.empty(result_matrix_shape, dtype=object)
    times = np.empty(result_matrix_shape, dtype=float)

    # Iterate through all results, storing results in the above arrays
    count, read, failed = 0, 0, 0
    for case_info in iterable:
        count += 1
        filename = utils.result_filename(METHOD_NAME, case_info)
        print('  processing "%s"' % filename)
        # Load the result
        try:
            result = utils.read_pickle(args.directory + filename)
            read += 1
        except Exception as e:
            print('    Could not open file "%s"' % filename)
            print("     ", e)
            continue
        # Process the result
        if isinstance(result, tuple) and isinstance(result[0], Exception):
            print('    WARNING - test case resulted in exception:', result)
            failed += 1
        else:
            # Store results into arrays
            idx = case_info_to_indices(case_info)
            # Common results
            estimates[idx] = result['estimate']
            I_estimates[idx] = result['estimated_I']
            times[idx] = result['elapsed']
            print("  done")

    # Store compiled results
    results = {'estimates': estimates,
               'I_estimates': I_estimates,
               'times': times}
    path = args.directory + utils.compiled_results_filename(METHOD_NAME)
    utils.write_pickle(path, ((args, gies_options, lmbdas, Ns), results))
    print('\nProcessed %d/%d - read %d/%d results - %d/%d results were an exception' %
          (count, n_samples, read, count, failed, count))
    print('Wrote compiled results to "%s"' % path)


# --------------------------------------------------------------------
# Execute experiments

def worker(case_info):
    # Run method
    try:
        result = run_method(case_info, debug=args.debug)
    except Exception as e:
        trace = traceback.format_exc()
        print("ERROR:", trace)
        result = (e, trace)
    filename = utils.result_filename(METHOD_NAME, case_info)
    utils.write_pickle(args.directory + filename, result)
    print("    Stored result in", filename) if args.debug else None


if not args.compile_only:
    n_workers = os.cpu_count() - 1 if args.n_workers == -1 else args.n_workers
    print("\n\nBeginning execution of %d experiments with %d workers at %s\n\n" %
          (len(iterable), n_workers, datetime.now()))
    start = time.time()
    if n_workers == 1:
        # Without multiprocessing, i.e. map function runs sequentially
        print("Running experiments sequentially")
        list(map(worker, iterable))
    else:
        # Or in parallel on a pool of n_workers
        print("Running experiments in parallel")
        with multiprocessing.Pool(n_workers) as pool:
            pool.map(worker, iterable, chunksize=args.chunksize)

    end = time.time()
    print("\n\nFinished experiments at %s (elapsed %0.2f seconds)\n\n" %
          (datetime.now(), end - start))

process_results()
