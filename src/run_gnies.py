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

METHOD_NAME = 'gnies'

# --------------------------------------------------------------------
# Auxiliary functions
if __name__ != '__main__':
    msg = "run_gnies.py should be run as a python script, e.g.\npython -m ut_lvce.comparison_experiments <args>."
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
    # GnIES parameters
    'backward_phase': {'default': False, 'type': bool},
    'fit_intercept': {'default': False, 'type': bool},
    'ges_one_run': {'default': False, 'type': bool},
    'ges_phases': {'default': 'fbt', 'type': str},
    'lambda_lo': {'default': 0.5, 'type': float},
    'lambda_hi': {'default': 0.5, 'type': float},
    'lambda_delta': {'default': 1e-3, 'type': float},
    'gnies_verbose': {'default': False, 'type': bool},
    'store_history': {'default': False, 'type': bool},
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
# Run algorithm on samples

print("Visible workers: %d vs. args.n_workers %d" %
      (os.cpu_count(), args.n_workers))

# Extract dataset information
info = utils.read_pickle(args.directory + utils.INFO_FILENAME)
n_cases, runs, Ns, p = info['n_cases'], info['runs'], info['Ns'], info['args'].p
n_samples = n_cases * runs * len(Ns)
print("Dataset contains a total of %d samples from %d cases at sample sizes %s for %d runs" %
      (n_samples, n_cases, Ns, runs))
# Build iterable
iterable = gnies.utils.cartesian([Ns, range(n_cases), range(runs)], dtype=int)
# print(iterable)
assert len(iterable) == n_samples

# Code to run GnIES
ges_phases = []
for symbol in args.ges_phases:
    if symbol == 'f':
        ges_phases.append('forward')
    elif symbol == 'b':
        ges_phases.append('backward')
    elif symbol == 't':
        ges_phases.append('turning')
    else:
        raise ValueError('Invalid symbol "%s" in args.ges_phases ("%s").' %
                         (symbol, args.ges_phases))

gnies_options = {'backward_phase': args.backward_phase,
                 'centered': not args.fit_intercept,
                 'ges_iterate': not args.ges_one_run,
                 'ges_phases': ges_phases,
                 'debug': args.gnies_verbose
                 }

print("Running GnIES with settings:")
print(" ", gnies_options)


def run_gnies(lmbda, n, case, run, debug=False):
    # Load data
    data_path = args.directory + utils.test_case_filename(n, case, run)
    data = utils.load_bin(data_path)
    N = sum([len(X) for X in data])
    lmbda = lmbda * np.log(N)
    # Run method
    start = time.time()
    score, estimated_icpdag, estimated_I, history = gnies.fit(data, **gnies_options)
    elapsed = time.time() - start
    print("  Ran GnIES on test case n:%d_g:%d_r:%d in %0.2f seconds." %
          (n, case, run, elapsed))
    # Store results
    result = {'estimate': estimated_icpdag,
              'estimated_I': estimated_I,
              'score': score,
              'elapsed': elapsed}
    if args.store_history:
        result['history'] = history
    return result


def process_results():
    print("Processing results:\n")
    # Common to all methods
    estimates = np.empty((len(Ns), n_cases, runs, p, p), dtype=int)
    size = (len(Ns), n_cases, runs)
    I_estimates = np.empty(size, dtype=object)
    times = np.empty(size, dtype=float)
    # For GnIES
    scores = np.empty(size, dtype=float)
    if args.store_history:
        history = np.empty(size, dtype=object)

    # Iterate through all results, storing results in the above arrays
    Ns_idx = dict(zip(sorted(Ns), range(len(Ns))))
    count, read, failed = 0, 0, 0
    for (n, case, run) in iterable:
        count += 1
        filename = utils.result_filename(METHOD_NAME, n, case, run)
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
            i = Ns_idx[n]
            estimates[i, case, run, :, :] = result['estimate']
            I_estimates[i, case, run] = result['estimated_I']
            times[i, case, run] = result['elapsed']
            scores[i, case, run] = result['score']
            if args.store_history:
                history[i, case, run] = result['history']
            print("  done")

    # Store compiled results
    results = {'estimates': estimates,
               'I_estimates': I_estimates,
               'times': times,
               'scores': scores}
    if args.store_history:
        results['history'] = history
    path = args.directory + utils.compiled_results_filename(METHOD_NAME)
    utils.write_pickle(path, results)
    print('\nProcessed %d/%d - read %d/%d results - %d/%d results were an exception' %
          (count, n_samples, read, count, failed, count))
    print('Wrote compiled results to "%s"' % path)


def worker(case_tuple):
    n, case, run = case_tuple
    # Run method
    try:
        result = run_gnies(n, case, run, debug=args.debug)
    except Exception as e:
        trace = traceback.format_exc()
        print("ERROR:", trace)
        result = (e, trace)
    path = args.directory + utils.result_filename(METHOD_NAME, n, case, run)
    utils.write_pickle(path, result)
    print("    Stored result in", path) if args.debug else None


n_workers = os.cpu_count() - 1 if args.n_workers == -1 else args.n_workers
print("\n\nBeginning experiments with %d workers on %d cases at %s\n\n" %
      (n_workers, n_samples, datetime.now()))
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

# --------------------------------------------------------------------
# Process results

# Extract estimates

# # Save arguments, test cases and compiled results
# filename = args.directory + 'compiled_results_gnies.pickle'
# with open(filename, 'wb') as f:
#     pickle.dump((args, test_cases, results), f)
#     print('\nWrote compiled results to "%s"' % filename)
