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

from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
from causaldag import unknown_target_igsp

import gnies.utils as utils

# --------------------------------------------------------------------
# TODO: icpdag=True here conflicts with the call from run_ut_igsp, refactor

def fit(data, alpha_ci, alpha_inv, debug=0):
    observational_sample = data[0]
    interventional_samples = data[1:]
    p = observational_sample.shape[1]
    nodes = set(range(p))
    # Form sufficient statistics
    ci_suffstat = gauss_ci_suffstat(observational_sample)
    invariance_suffstat = gauss_invariance_suffstat(observational_sample,
                                                    interventional_samples)
    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(gauss_ci_test,
                                  ci_suffstat,
                                  alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test,
                                                 invariance_suffstat,
                                                 alpha=alpha_inv)
    # Run UT-IGSP
    setting_list = [dict(known_interventions=[])] * (len(data) - 1)
    estimated_dag, est_targets_list = unknown_target_igsp(
        setting_list, nodes, ci_tester, invariance_tester)
    # Process estimates
    estimated_dag = estimated_dag.to_amat()[0]
    estimated_I = set.union(*est_targets_list)
    estimated_icpdag = utils.dag_to_icpdag(estimated_dag, estimated_I)
    return (estimated_icpdag, estimated_I, estimated_dag)
