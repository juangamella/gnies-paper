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

import numpy as np
import gnies.utils
import src.utils as utils
import argparse
import os
import causalchamber
import pandas as pd
import time

# --------------------------------------------------------------------
# Variable names

# Mapping from experiments -> environments
environments = [
    ["uniform_reference"],  # reference environment
    ["uniform_red_mid", "uniform_red_strong"],  # intervention on R
    ["uniform_green_mid", "uniform_green_strong"],  # intervention on G
    ["uniform_blue_mid", "uniform_blue_strong"],  # intervention on B
    ["uniform_pol_1_mid", "uniform_pol_1_strong"],  # intervention on \theta_1
    ["uniform_pol_2_mid", "uniform_pol_2_strong"],  # intervention on \theta_2
    ["uniform_t_ir_1_strong"],
    ["uniform_t_ir_2_strong"],
    ["uniform_t_ir_3_strong"],
    ["uniform_osr_angle_1_strong"],  # intervention on \tilde{\theta}_1 (through O_1)
    ["uniform_osr_angle_2_strong"],  # intervention on \tilde{\theta}_2 (through O_2)
]

variables = [
    "red",
    "green",
    "blue",
    "current",
    "ir_1",
    "ir_2",
    "ir_3",
    "pol_1",
    "angle_1",
    "pol_2",
    "angle_2",
]


def load_dataset():
    # Download dataset
    dataset = causalchamber.datasets.Dataset(
        "lt_interventions_standard_v1", root="/tmp"
    )
    # Select and merge experiments into environments (defined above)
    dataframes = [
        pd.concat(
            [dataset.get_experiment(f).as_pandas_dataframe() for f in F],
            ignore_index=True,
        )
        for F in environments
    ]
    return dataframes


def prepare_experiments_directory(path, Ns, runs):
    # Load ground-truth graph
    graph = (
        causalchamber.ground_truth.graph("lt", "standard")
        .loc[variables][variables]
        .values
    )
    # Prepare and save test case info
    args = argparse.Namespace()
    args.p = len(variables)
    to_save = {
        "n_cases": 1,
        "cases": [graph],
        "runs": runs,
        "Ns": Ns,
        "args": args,
        "graph": graph,
        "variables": variables,
    }
    # Write test case info
    path += "" if path[-1] == "/" else "/"
    directory = path + "dataset_%d_light_tunnel/" % time.time()
    os.makedirs(directory)
    filename = directory + utils.INFO_FILENAME
    utils.write_pickle(filename, to_save)
    print('  saved test case info to "%s"' % filename)
    # Write test_case_graph
    filename = directory + "graph"
    utils.data_to_bin(graph, filename, debug=True)
    print('  saved graph to "%s"' % filename)
    # Write data
    dataframes = load_dataset()
    for n in Ns:
        for r in np.arange(runs):
            data = [
                df[variables].sample(n=n, random_state=r).values for df in dataframes
            ]
            data = utils.standardize(data)
            filename = directory + utils.test_case_filename(n, 0, r)
            utils.data_to_bin(data, filename, debug=True)
