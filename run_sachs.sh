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

# Commands to run the experiments on the Sachs dataset

# Note: UT-IGSP was run with the regularization parameters used for
# this dataset in their paper, see line 40 in
# https://github.com/csquires/utigsp/blob/master/real_data_analysis/sachs/sachs_run_algs.py


# ----------------------------------------------------------------------
# Figure 3, left: Experiments on the raw dataset from Sachs et al. 2005

RAW_DATASET="sachs_experiments/dataset_1661368400_sachs_sachs_2005_normalized:True"
# The dataset was generated by running the following python code from
# the root of the directory (where this file is contained):
#   >>> import sachs.sachs_2005 as sachs
#   >>> sachs.prepare_experiments_directory('sachs_experiments/', 'consensus', normalize=True)

# Run GnIES, GES
python3 -m src.run_gnies --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $RAW_DATASET
python3 -m src.run_gnies --rank --phases f --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $RAW_DATASET
python3 -m src.run_gnies --rank --phases b --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $RAW_DATASET
python3 -m src.run_ges --lambdas 0.01,0.25,0.5,0.75,1,2 --n_workers 1 --directory $RAW_DATASET

# Run UT-IGSP with Gaussian tests and a different observational environment
python3 -m src.run_utigsp --n_workers 1 --obs 0 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 1 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 2 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 3 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 4 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 5 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 6 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 7 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 8 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET

# Run UT-IGSP with Kernel tests and a different observational environment
python3 -m src.run_utigsp --n_workers 8 --obs 0 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 1 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 2 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 3 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 4 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 5 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 6 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 7 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 8 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $RAW_DATASET

./compute_metrics.sh $RAW_DATASET

# ----------------------------------------------------------------------
# # Figure 3, right: Experiments on the hybrid dataset

HYBRID_DATASET="sachs_experiments/dataset_1661774872_runs:10_seed:42_tag:sachs_2005_normalized_type:drf_standardize:0_n:None/"
# The dataset was generated using the command:
# python3 -m src.gen_hybrid_data --tag sachs_2005_normalized --type drf --runs 10 --directory sachs_experiments/ --dataset $RAW_DATASET/test_case_n:-1_g:0_r:0.npz --graph $RAW_DATASET/graph.npy --cluster

# Run GnIES, GES
python3 -m src.run_gnies --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $HYBRID_DATASET
python3 -m src.run_gnies --rank --phases f --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $HYBRID_DATASET
python3 -m src.run_gnies --rank --phases b --n_workers 5 --lambdas 0.01,0.25,0.5,0.75,1,2 --directory $HYBRID_DATASET
python3 -m src.run_ges --lambdas 0.01,0.25,0.5,0.75,1,2 --n_workers 1 --directory $HYBRID_DATASET

# Run UT-IGSP with Gaussian tests and a different observational environment
python3 -m src.run_utigsp --n_workers 1 --obs 0 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 1 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 2 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 3 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 4 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 5 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 6 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 7 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 1 --obs 8 --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET

# Run UT-IGSP with Kernel tests and a different observational environment
python3 -m src.run_utigsp --n_workers 8 --obs 0 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 1 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 2 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 3 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 4 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 5 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 6 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 7 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET
python3 -m src.run_utigsp --n_workers 8 --obs 8 --test hsic --alphas "1e-1,1e-2,1e-3,2e-1,3e-1,4e-1,5e-1,5e-2" --betas "1e-20" --directory $HYBRID_DATASET

./compute_metrics.sh $HYBRID_DATASET
