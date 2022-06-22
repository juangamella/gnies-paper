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
# - Random seed management
# - Verbosity

import pickle
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Definitions

INFO_FILENAME = 'test_cases.pickle'


# --------------------------------------------------------------------
# Auxiliary functions

def test_case_filename(n, graph, run):
    return 'test_case_n:%d_g:%d_r:%d' % (n, graph, run)


def result_filename(method, n, graph, run):
    return "result_%s_n:%d_g:%d_r:%d.pickle" % (method, n, graph, run)


def compiled_results_filename(method):
    return "compiled_results_%s.pickle" % method


def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(filename):
    if os.path.getsize(filename) > 0:
        with open(filename, "rb") as f:
            return pickle.load(f)


def load_bin(path):
    return np.load(path + '.npy', allow_pickle=False)


def data_to_csv(data, path, debug):
    """Save the multienvironment data in a csv file ready to be used by the R script that runs ICP"""
    # Combine the data into a single matrix with an extra column indicating the environment
    flattened = []
    for e, X in enumerate(data):
        flagged = np.hstack([np.ones((len(X), 1)) * e, X])
        flattened.append(flagged)
    flattened = np.vstack(flattened)
    # Save to .csv
    df = pd.DataFrame(flattened)
    filename = path + '.csv'
    print('  saved test case data to "%s"' % filename) if debug else None
    df.to_csv(filename, header=False, index=False)


def pooled_data_to_csv(data, path, debug, normalize=False):
    """Pool the multienvironment data and save into a single .csv file."""
    pooled = np.vstack(data)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    normalized = (pooled - mean) / std
    # Save to .csv
    df = pd.DataFrame(normalized)
    filename = path + '.csv'
    print('  saved pooled test case data to "%s"' %
          filename) if debug else None
    df.to_csv(filename, header=False, index=False)


def data_to_bin(data, path, debug):
    filename = path + '.npy'
    print('  saved test case data to "%s"' % filename) if debug else None
    np.save(path, data)


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

# --------------------------------------------------------------------
# Process results


def compile_results(directory, clean=False):
    print('Compiling test cases in "%s"' % directory)
    results = []
    for entry in os.scandir(directory):
        print('\n  Reading', entry.name)
        # Read test cases file
        if entry.name == 'test_cases.pickle':
            with open(entry.path, 'rb') as f:
                (namespace, test_cases) = pickle.load(f)
            print('    loaded test cases and namespace')
            print('      %s' % namespace)
        # Read individual test case file
        elif entry.name.startswith('test_case_'):
            wo_prefix = entry.name.split('test_case_')[1]
            case_id = int(wo_prefix.split('.pickle')[0])
            with open(entry.path, 'rb') as f:
                (idx, test_case, result) = pickle.load(f)
            results.append((idx, test_case, result))

            # Output summary
            if case_id != idx:
                print(
                    '    WARNING - case id mismatch, file says %s, stored says %s' % (case_id, idx))
            if isinstance(result, tuple) and isinstance(result[0], Exception):
                print(
                    '    WARNING - test case resulted in exception:', result[0])
                print(result[1])
            else:
                print('    processed result')
        # Ignore other files
        else:
            print('    ignoring')

    # Save compiled results
    filename = directory + '/' + 'compiled_results.pickle'
    write_pickle(filename, (namespace, test_cases, results))
    print('\nWrote compiled results to "%s"' % filename)
    return filename
