# Copyright 2021 Juan L Gamella

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

from jci.pc import PC

import gnies.utils as utils
import pandas as pd
import numpy as np

# --------------------------------------------------------------------


def fit(data, alpha, debug=0):
    # Transform data into the appropriate format, i.e. a single array
    # with the pooled data from all environments, and a 1-dim array
    # indicating to which "regime" (environment) each observation
    # belongs
    regimes = []
    for i, env in enumerate(data):
        regimes.append(np.ones(len(env), dtype=int) * (i+1))
    regimes = pd.DataFrame(np.hstack(regimes))
    pooled = pd.DataFrame(np.vstack(data))
    # Call JCI-PC
    instance = PC(verbose=True)
    dag = instance._run_pc(data=pooled, alpha=alpha, regimes=regimes)
    # Extract result
    p, e = data[0].shape[1], len(data)
    # I-CPDAG is [0,..,p] subgraph of the CPDAG of the returned graph
    estimated_cpdag = utils.dag_to_cpdag(dag)
    estimate = estimated_cpdag[0:p, 0:p]
    # Targets are system variables which are children of context variables
    context_variables = p + np.arange(e)
    estimated_I = set.union(*[utils.ch(i, dag) for i in context_variables]) - set(context_variables)
    return estimate, estimated_I, estimated_cpdag
