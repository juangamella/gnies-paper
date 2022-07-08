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

"""
"""

# TODO

import numpy as np
import time
from datetime import datetime
import argparse
import os
import multiprocessing
import gnies.utils
import src.utils as utils
import traceback
import src.jci_pc as jci_pc

METHOD_NAME = 'jci_pc'

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
    # UT-IGSP parameters
    'alpha_lo': {'default': 0.01, 'type': float},
    'alpha_hi': {'default': 0.01, 'type': float},
    'n_alphas': {'default': 1, 'type': int},
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
    'directory',
    'chunksize',
    'n_workers',
    'compile_only'
]

# --------------------------------------------------------------------
# Run algorithm on samples

print("Visible workers: %d vs. args.n_workers %d" %
      (os.cpu_count(), args.n_workers))

# Extract dataset information
info = utils.read_pickle(args.directory + utils.INFO_FILENAME)
n_cases, runs, Ns, p = info['n_cases'], info['runs'], info['Ns'], info['args'].p
n_samples = n_cases * runs * len(Ns)
print("Dataset contains a total of %d samples from %d cases at sample sizes %s for %d runs" %
      (n_samples, n_cases, Ns, runs))

# --------------------------------------------------------------------

# ---------------------------------------------------------------------
# Build iterable, result_matrix_shape, iterable entry to index function

# Significance level for CI tests
alphas = utils.hyperparameter_range(args.alpha_lo,
                                    args.alpha_hi,
                                    args.n_alphas)

fields = [range(n_cases), alphas, Ns, range(runs)]

iterable = []
for (graph, alpha, sample_size, run) in gnies.utils.cartesian(fields, dtype=object):
    iterable.append({'a': alpha,
                     'n': sample_size,
                     'g': graph,
                     'r': run})
assert len(iterable) == n_samples * len(alphas)

# NOTE!!!: The case must be the first index (to easily recover ground
# truth when computing metrics)

result_matrix_shape = tuple(len(field) for field in fields)

Ns_idx = dict(zip(sorted(Ns), range(len(Ns))))
alphas_idx = dict(zip(sorted(alphas), range(len(alphas))))


def case_info_to_indices(info):
    alpha = info['a']
    n = info['n']
    return (info['g'], alphas_idx[alpha], Ns_idx[n], info['r'])


# ------------------
# Code to run method

print("Running JCI-PC with Gaussian CI tests")
print("  on alphas :", alphas)


def run_method(info, debug=False):
    """Takes an iterable entry and runs the algorithm accordingly"""
    # Load data
    n, graph, run = info['n'], info['g'], info['r']
    data_path = args.directory + utils.test_case_filename(n, graph, run)
    data = utils.load_bin(data_path)
    # Run method
    start = time.time()
    output = jci_pc.fit(data, alpha=info['a'])
    elapsed = time.time() - start
    print("  Ran JCI-PC on test case %s in %0.2f seconds." %
          (utils.serialize_dict(info), elapsed)) if debug else None
    # Store results
    estimate, estimated_I, estimated_full_cpdag = output
    result = {'estimate': estimate,
              'estimated_I': estimated_I,
              'estimated_full_cpdag': estimated_full_cpdag,
              'alpha': info['a'],
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
    utils.write_pickle(path, ((args, alphas, Ns), results))
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
