# Copyright 2020 Juan L Gamella

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
# - Random seed management
# - Decide how to divide results / run (probably each method-wise)

import pickle
import time
from datetime import datetime
import argparse
import os
import numpy as np
import sempler
import sempler.generators
import multiprocessing

# For GES variants

from ges.scores.experimental import FixedInterventionalScore, ExperimentalScore
import ges.test_neighborness as test_neighborness
import ges.test_orient_greediness as test_orient
import ges.alternating_ut_ges as alternating_ut_ges
import ges.ges as ges

# For UT-IGSP

# When the causaldag module is imported, multiprocessing fails with error:

# _pickle.PicklingError: Can't pickle <function wrapper at 0x7f29919e3ea0>: attribute lookup wrapper on __main__ failed

# If causaldag is imported in the __init__.py of the package, the
# problem disappears. It also does not ocurr when executing the module
# as a script.

import ges.ut_igsp as ut_igsp

# ---------------------------------------------------------------------------------------------------
# Test case generation + evaluation functions


def gen_scms(G, p, k, w_min, w_max, m_min, m_max, v_min, v_max):
    """
    Generate random experimental cases (ie. linear SEMs). Parameters:
      - n: total number of cases
      - p: number of variables in the SCMs
      - k: average node degree
      - w_min, w_max: Weights of the SCMs are sampled at uniform between w_min and w_max
      - v_min, v_max: Variances of the variables are sampled at uniform between v_min and v_max
      - m_min, m_max: Intercepts of the variables of the SCMs are sampled at uniform between m_min and m_max
      - random_state: to fix the random seed for reproducibility
    """
    cases = []
    while len(cases) < G:
        W = sempler.generators.dag_avg_deg(p, k, w_min, w_max)
        scm = sempler.LGANM(W, (m_min, m_max), (v_min, v_max))
        cases.append(scm)
    return cases


def generate_interventions(scm, no_ints, max_int_size, m_min, m_max, v_min, v_max, include_obs=True):
    """Generate a set of interventions for a given scm, randomly sampling
    no_ints sets of targets of size int_size, and sampling the
    intervention means/variances uniformly from [m_min,m_max], [v_min, v_max].

    If include_obs is True, include an empty intervention to represent
    the reference or observational environment.
    """
    interventions = [None] if include_obs else []
    int_size = max_int_size  # np.random.randint(0, max_int_size + 1)
    union_of_targets = set()
    # For each intervention
    for _ in range(no_ints):
        # sample targets
        base = set(range(scm.p)) - union_of_targets
        targets = np.random.choice(list(base), int_size, replace=False)
        union_of_targets |= set(targets)
        # sample parameters
        means = np.random.uniform(m_min, m_max, len(targets)) if m_min != m_max else [
            m_min] * len(targets)
        variances = np.random.uniform(v_min, v_max, len(
            targets)) if v_min != v_max else [v_min] * len(targets)
        # assemble intervention
        intervention = dict((t, (mean, var)) for (t, mean, var) in zip(targets, means, variances))
        interventions.append(intervention)
    return union_of_targets, interventions


def evaluate_methods(test_case, methods, n, n_obs, int_type, debug=False):
    """Evaluate the given methods on the given test case"""
    (scm, interventions, union_of_targets) = test_case
    print("TEST CASE:", interventions, union_of_targets)
    true_I = union_of_targets
    true_icpdag = ges.utils.dag_to_imec(scm.W, true_I)
    # Sample interventional data
    XX = []
    population_covariances = []
    for intervention in interventions:
        if intervention is None:
            X = scm.sample(n=n_obs)
            cov = scm.sample(population=True).covariance
        elif int_type == 'shift':
            X = scm.sample(n=n, shift_interventions=intervention)
            cov = scm.sample(population=True, shift_interventions=intervention).covariance
        elif int_type == 'do':
            X = scm.sample(n=n, do_interventions=intervention)
            cov = scm.sample(population=True, do_interventions=intervention).covariance
        else:
            raise ValueError("Invalid intervention type: %s" % int_type)
        XX.append(X)
        population_covariances.append(cov)
    # Evaluate methods on sampled data
    truth = {'estimate': true_icpdag, 'I': true_I, 'time': 0}
    results = {'truth': truth}
    debug_string = "Case finished. Runtimes (seconds)"
    for (method_name, fun) in methods.items():
        # Run method
        start = time.time()
        result = fun(XX, population_covariances, true_I)
        end = time.time()
        # Store elapsed time
        result['time'] = end - start
        results[method_name] = result
        # Check if matches
        icpdag = result['estimate']
        I = result['I']
        matches = (icpdag == true_icpdag).all() and I == true_I
        debug_string += " -- " + method_name + \
            " (matches : %s)" % matches + " %0.2f" % (end - start)
    print(debug, debug_string)
    return results

# --------------------------------------------------------------------
# Parse input parameters


# Definitions and default settings
arguments = {
    # Execution parameters
    'n_workers': {'default': 1, 'type': int},
    # 'batch_size': {'default': 20000, 'type': int},
    'runs': {'default': 1, 'type': int},
    'random_state': {'default': 42, 'type': int},
    'tag': {'type': str},
    'debug': {'default': False, 'type': bool},
    'chunksize': {'type': int, 'default': 1},
    # SCM generation parameters
    'G': {'default': 1, 'type': int},
    'k': {'default': 2.7, 'type': float},
    'p': {'default': 5, 'type': int},
    'w_min': {'default': 0.5, 'type': float},
    'w_max': {'default': 1, 'type': float},
    'v_min': {'default': 1, 'type': float},
    'v_max': {'default': 2, 'type': float},
    'm_min': {'default': -1, 'type': float},
    'm_max': {'default': 1, 'type': float},
    # Intervention parameters
    'int_type': {'default': 'shift', 'type': str},
    'int_size': {'default': 1, 'type': int},
    'no_ints': {'default': 3, 'type': int},
    'int_m_min': {'default': 0, 'type': float},
    'int_m_max': {'default': 0, 'type': float},
    'int_v_min': {'default': 5, 'type': float},
    'int_v_max': {'default': 10, 'type': float},
    # Sampling parameters
    'n': {'default': 1000, 'type': int},
    'n_obs': {'type': int},
    'pop': {'default': False, 'type': bool},
    # UT-GES parameters
    'iterate': {'default': False, 'type': bool},
    'phases': {'type': str}
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

# Parameters that will be excluded from the filename (see parameter_string function above)
excluded_keys = ['debug', 'n_workers', 'chunksize']  # , 'batch_size']
excluded_keys += ['tag'] if args.tag is None else []
excluded_keys += ['n_obs'] if args.n_obs is None else []

print(args)  # For debugging

# Set random seed
np.random.seed(args.random_state)

# --------------------------------------------------------------------
# Generate test cases
#   A test case is an SCM and a set of interventions

print("Generating SCMs...", end="")
SCMs = gen_scms(args.G,
                args.p,
                args.k,
                args.w_min,
                args.w_max,
                args.m_min,
                args.m_max,
                args.v_min,
                args.v_max)
print(" done.")

cases = []
print("Generating interventions...", end="")
for scm in SCMs:
    union_of_targets, interventions = generate_interventions(scm,
                                                             args.no_ints,
                                                             args.int_size,
                                                             args.int_m_min,
                                                             args.int_m_max,
                                                             args.int_v_min,
                                                             args.int_v_max)
    cases.append((scm, interventions, union_of_targets))
print(" done.")
# --------------------------------------------------------------------
# Experiments

# Parse phases (for ut-ges)
if args.phases is None:
    phases = ['forward', 'turn', 'backward']
else:
    phases = []
    for char in args.phases:
        if char == 'f':
            phases.append('forward')
        elif char == 't':
            phases.append('turn')
        elif char == 'b':
            phases.append('backward')
        else:
            raise ValueError('Invalid phase specified: %s' % args.phases)

# Define algorithms


def GES_pooled_w_orient(XX, population_covariances=None, targets=None):
    """Run GES with joint observational score and then shrink down the
    CPDAG using the orient operator"""
    start = time.time()
    print("GES with joint obs. score + orient operator") if args.debug else None
    # Run GES with the multi-environment observational score
    # (i.e. over the pooled, centered data)
    obs_score = FixedInterventionalScore(XX, set())
    if args.pop:
        obs_score._sample_covariances = population_covariances
    cpdag, _ = ges.fit(obs_score)
    print("  estimated CPDAG") if args.debug else None
    # Narrow the equivalence class down using the orient operator
    int_score = ExperimentalScore(XX, fine_grained=False, centered=True)
    _, icpdag, I = test_orient.max_scoring_orient_half_greedy(cpdag, set(), int_score)
    print("  estimated I-CPDAG") if args.debug else None
    print("Done (%0.2f seconds)\n" % (time.time() - start)) if args.debug else None
    return {'estimate': icpdag, 'I': I, 'cpdag': cpdag}


def GES_pooled(XX, population_covariances, targets=None):
    start = time.time()
    print("GES with joint obs. score + orient operator") if args.debug else None
    # Run GES with the multi-environment observational score
    # (i.e. over the pooled, centered data)
    obs_score = FixedInterventionalScore(XX, set())
    if args.pop:
        obs_score._sample_covariances = population_covariances
    cpdag, _ = ges.fit(obs_score, iterate=args.iterate)
    print("Done (%0.2f seconds)\n" % (time.time() - start)) if args.debug else None
    return {'estimate': cpdag, 'I': set(), 'cpdag': cpdag}


def GES_fixed_I(XX, population_covariances, targets):
    start = time.time()
    print("GES with fixed targets: %s" % targets) if args.debug else None
    score_class = FixedInterventionalScore(XX, targets)
    if args.pop:
        score_class._sample_covariances = population_covariances

    def completion_algorithm(P):
        return ges.utils.pdag_to_imec(P, targets)
    est, score = ges.fit(score_class, completion_algorithm, iterate=True)
    print("Done (%0.2f seconds)\n" % (time.time() - start)) if args.debug else None
    return {'estimate': est, 'I': targets, 'score': score}

def alternating_UT_GES_backward(XX, population_covariances, targets, backward_phase=False):
    return alternating_UT_GES(XX, population_covariances, targets, backward_phase=True)

def alternating_UT_GES(XX, population_covariances, targets, backward_phase=False):
    start = time.time()
    print("Alternating GES") if args.debug else None
    population_covariances = np.array(population_covariances) if args.pop else None
    score, estimate, I, history = alternating_ut_ges.fit(XX, population_covariances, backward_phase=backward_phase)
    print("Done (%0.2f seconds)\n" % (time.time() - start)) if args.debug else None
    return {'estimate': estimate, 'I': I, 'history': history, 'score': score}


def ut_igsp_wrapper(XX, pipulation_covariances=None, targets=None):
    """Run UT-IGSP on a collection of samples"""
    (icpdag, I) = ut_igsp.fit(XX, debug=args.debug)
    return {'estimate': icpdag, 'I': I}


def ut_ges(XX, population_covariances=None, targets=None):
    coarse_score = ExperimentalScore(XX, fine_grained=False, centered=True)
    if args.pop:
        coarse_score._sample_covariances = population_covariances
    result = test_neighborness.brute_force_ut_ges(coarse_score,
                                                  phases=phases,
                                                  iterate=args.iterate,
                                                  debug=2 if args.debug else 0)
    (icpdag, I, _), history = result
    return {'estimate': icpdag, 'I': I, 'history': history}


# Setup
methods = {  # 'GES-cdt': GES_cdt,
    'alternating_UT_GES': alternating_UT_GES,
    'alternating_UT_GES_w_backward_phase': alternating_UT_GES_backward,
    # 'GES-pooled': GES_pooled,
    'UT-IGSP': ut_igsp_wrapper,
    # # 'UT-GES': ut_ges,
    'GES-true-I': GES_fixed_I
}

n_workers = os.cpu_count() - 1 if args.n_workers == -1 else args.n_workers


def wrapper(test_case):
    return evaluate_methods(test_case,
                            methods,
                            args.n,
                            args.n_obs if args.n_obs is not None else args.n,
                            args.int_type,
                            debug="  " if args.debug else False)


iterator = cases * args.runs

# Run experiments
start = time.time()
print("\n\nBeginning experiments with %d workers on %d cases x %d runs = %d experiments at %s\n\n" %
      (n_workers, len(cases), args.runs, len(iterator), datetime.now()))
if n_workers == 1:
    pooled_results = list(map(wrapper, iterator))
else:
    pool = multiprocessing.Pool(n_workers)
    pooled_results = pool.map(wrapper, iterator, chunksize=args.chunksize)
end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end - start))

# --------------------------------------------------------------------
# Save results

# Process results
method_times = {}
results = {}
# 1. Group method-wise
for result in pooled_results:
    for (method_name, method_result) in result.items():
        # Store result
        current = results.get(method_name, [])
        results[method_name] = current + [method_result]
        # Accumulate total elapsed time per method
        current_total_time = method_times.get(method_name, 0)
        method_times[method_name] = current_total_time + method_result['time']

# 2. Partition into runs
for (method_name, method_results) in results.items():
    idx = range(0, len(cases) * args.runs, len(cases))
    by_runs = [method_results[i:i + len(cases)] for i in idx]
    results[method_name] = by_runs

if args.debug:
    assert len(results) == len(methods) + 1
    for method in methods.keys():
        assert len(results[method]) == args.runs
        assert len(results[method][0]) == len(cases)

# Display total method times
print("Average time/experiment (seconds)-", end="")
for (method, total_time) in method_times.items():
    print(" %s: %0.2f" % (method, total_time / len(iterator)), end="")
print()

# Compose filename


def parameter_string(args, excluded_keys):
    """Convert a Namespace object (from argparse) into a string, excluding
    some keys, to use as filename or dataset name"""
    string = ""
    for k, v in vars(args).items():
        if isinstance(v, bool):
            value = str(int(v))
        else:
            value = str(v)
        value = value.replace('/', '')
        if k not in excluded_keys:
            string = string + "_" + k + ":" + value
    return string


filename = "experiments/results_%d" % end
filename = filename + parameter_string(args, excluded_keys) + ".pickle"

# Pickle results
to_pickle = (args, cases, results)
print(to_pickle[0])
pickle.dump(to_pickle, open(filename, "wb"))
print("Saved to file \"%s\"" % filename)
