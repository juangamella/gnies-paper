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

# Commands to run the synthetic experiments


# --------------------------------------------------------------------
# Figure 1: Model match, standardized data.

DATASET=
# The dataset was generated using the command:
# python3 -m src.generate_synthetic_data --cluster --G 100 --runs 10 --n 10,100,1000 --i_size 1 --e 4 --p 10 --i_type noise --obs --seed 42 --tag I3

# Run the methods: GnIES, UT-IGSP, GES and sortnregress
bsub -n 50 -W 30:00 python3 -m src.run_gnies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_utigsp --n_workers 49 --alpha_lo 0.00001 --alpha_hi 0.1 --n_alphas 5 --beta_lo 0.00001 --beta_hi 0.1 --n_betas 5 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_ges --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:10 python3 -m src.run_sortnregress --n_workers 49 --pool --tag pool --directory $DATASET


# --------------------------------------------------------------------
# Figure 2: Model mismatch, standardized data.

DATASET=
# The dataset was generated using the command:
# python3 -m src.generate_synthetic_data --cluster --G 100 --runs 10 --n 10,100,1000 --i_size 1 --e 4 --p 10 --i_type do --obs --seed 42 --tag I3sd

# Run the methods: GnIES, UT-IGSP, GES, GIES and sortnregress
bsub -n 50 -W 30:00 python3 -m src.run_gnies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_utigsp --n_workers 49 --alpha_lo 0.00001 --alpha_hi 0.1 --n_alphas 5 --beta_lo 0.00001 --beta_hi 0.1 --n_betas 5 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_ges --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_gies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:10 python3 -m src.run_sortnregress --n_workers 49 --pool --tag pool --directory $DATASET


# --------------------------------------------------------------------
# Figure 6 (Appendix D): Comparison with raw (unstandardized) data

# -----------
# Model match

DATASET=
# The dataset was generated using the command:
# python3 -m src.generate_synthetic_data --cluster --G 100 --runs 10 --n 10,100,1000 --i_size 1 --e 4 --p 10 --i_type noise --obs --standardize --seed 42 --tag I3s

# Run the methods: GnIES, UT-IGSP, GES and sortnregress
bsub -n 50 -W 30:00 python3 -m src.run_gnies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_utigsp --n_workers 49 --alpha_lo 0.00001 --alpha_hi 0.1 --n_alphas 5 --beta_lo 0.00001 --beta_hi 0.1 --n_betas 5 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_ges --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:10 python3 -m src.run_sortnregress --n_workers 49 --pool --tag pool --directory $DATASET


# --------------
# Model mismatch

DATASET=
# The dataset was generated using the command:
# python3 -m src.generate_synthetic_data --cluster --G 100 --runs 10 --n 10,100,1000 --i_size 1 --e 4 --p 10 --i_type do --obs --standardize --seed 42 --tag I3sd

# Run the methods: GnIES, UT-IGSP, GES, GIES and sortnregress
bsub -n 50 -W 30:00 python3 -m src.run_gnies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_utigsp --n_workers 49 --alpha_lo 0.00001 --alpha_hi 0.1 --n_alphas 5 --beta_lo 0.00001 --beta_hi 0.1 --n_betas 5 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_ges --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:30 python3 -m src.run_gies --n_workers 49 --lambdas 0.01,0.25,0.5,0.75,1 --directory $DATASET
bsub -n 50 -W 0:10 python3 -m src.run_sortnregress --n_workers 49 --pool --tag pool --directory $DATASET
