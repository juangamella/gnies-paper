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
import gc  # Garbage collector
import time
import os
import pickle

# --------------------------------------------------------------------
# Parse input parameters

# Definitions and default settings
arguments = {
    # Execution parameters
    'cluster': {'default': False, 'type': bool},
    'runs': {'default': 1, 'type': int},
    'seed': {'default': 42, 'type': int},
    'tag': {'type': str},
    'debug': {'default': False, 'type': bool},
    # SCM generation parameters
    'G': {'default': 1, 'type': int},
    'k': {'default': 2.7, 'type': float},
    'p': {'default': 5, 'type': int},
    'w_min': {'default': 0.5, 'type': float},
    'w_max': {'default': 1, 'type': float},
    'v_min': {'default': 1, 'type': float},
    'v_max': {'default': 2, 'type': float},
    'm_min': {'default': 0, 'type': float},
    'm_max': {'default': 0, 'type': float},
    # Intervention parameters
    'envs': {'default': 3, 'type': int},
    'i_type': {'default': 'noise', 'type': str},
    'i_size': {'default': 1, 'type': int},
    'i_m_min': {'default': 0, 'type': float},
    'i_m_max': {'default': 0, 'type': float},
    'i_v_min': {'default': 5, 'type': float},
    'i_v_max': {'default': 10, 'type': float},
    # Sampling parameters
    'n': {'default': "1000", 'type': str},
    #    'n_obs': {'type': str},
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
excluded_keys = ['debug', 'cluster']  # , 'batch_size']
excluded_keys += ['tag'] if args.tag is None else []
excluded_keys += ['i_m_min', 'i_m_max'] if args.i_m_min == 0 and args.i_m_max == 0 else []
excluded_keys += ['m_min', 'm_max'] if args.m_min == 0 and args.m_max == 0 else []
#excluded_keys += ['n_obs'] if args.n_obs is None else []

print(args)  # For debugging


# --------------------------------------------------------------------
# Generate data for each test case

directory = "synthetic_experiments/dataset_%d%s/" % (time.time(),
                                                     utils.parameter_string(args, excluded_keys))

# If running on the Euler cluster, store results in the scratch directory
if args.cluster:
    directory = "/cluster/scratch/gajuan/" + directory

os.makedirs(directory)
print('Storing results in "%s"' % directory)


int_types = {'noise': 'noise_interventions',
             'shift': 'shift_interventions',
             'do': 'do_interventions'}

# Generate the SCMs + their interventions
rng = np.random.default_rng(args.seed)


def int_mean():
    return rng.uniform(args.i_m_min, args.i_m_max)


def int_var():
    return rng.uniform(args.i_v_min, args.i_v_max)


test_cases = []
for i in range(args.G):
    # Generate SCM
    W = gen.dag_avg_deg(args.p, args.k, args.w_min, args.w_max, random_state=args.seed + i)
    scm = sempler.LGANM(W, (args.m_min, args.m_max), (args.v_min, args.v_max))
    # Generate interventions and their parameters
    # interventions = [{int_types[args.i_type]: None}] if args.n_obs is not None else []
    interventions = []
    K = args.envs  # - len(interventions)  # number of "interventional" environments
    all_targets = gen.intervention_targets(
        args.p, K, args.i_size, replace=False, random_state=args.seed + i)
    for env_targets in all_targets:
        intervention = {int_types[args.i_type]:
                        dict((target, (int_mean(), int_var())) for target in env_targets)}
        interventions.append(intervention)
    test_cases.append((scm, interventions))

# Save test cases for analysis
Ns = [int(n) for n in args.n.split(',')]
to_save = {'n_cases': len(test_cases),
           'runs': args.runs,
           'Ns': Ns,
           'args': args,
           'cases': test_cases}
filename = directory + utils.INFO_FILENAME
with open(filename, 'wb') as f:
    pickle.dump(to_save, f)
    print('\nSaved test cases + info to "%s"' % filename)


# Sample data for each run


# obs_Ns = None if args.n_obs is None else [int(n) for n in args.n_obs.split('i')]
# assert (obs_Ns is None or len(Ns) == len(obs_Ns))


for n in Ns:
    print()
    print("---------------------------------")
    print("Generating datasets for n=%d" % n)
    for i, (scm, interventions) in enumerate(test_cases):
        print("  Test case %d" % i)
        print("    interventions:", interventions)
        print("    adjacency:")
        for row in scm.W:
            print(' ' * 5, (row != 0).astype(int))
        print("    generating data for run:")
        print(' ' * 5, end='')
        for j in range(args.runs):
            print(j, end=' ')
            # Generate the data for each environment
            data = []
            for intervention in interventions:
                data += [scm.sample(n, **intervention, random_state=j)]
            # Save the data to file
            path = directory + utils.test_case_filename(n, i, j)
            utils.data_to_bin(data, path, debug=args.debug)
            del data
            gc.collect()
        print()
