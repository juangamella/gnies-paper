# Experiment Repository for *"Characterization and greedy learning of Gaussian structural causal models under unknown interventions"*

This repository contains the code to reproduce the experiments and figures for the paper *"Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions"*, by JL. Gamella, A. Taeb, C. Heinze-Deml and P. Bühlmann. This README is not intended to be completely self-explanatory, and should be read alongside the manuscript (<TODO: link to arxiv>).

## Software packages

This repository contains only the code to reproduce the results from the paper. If you're interested in using the GnIES algorithm described in the paper for your own work, it is available as a separate and well-documented python package called `gnies`. You can find more information on its own repository at [github.com/juangamella/gnies](https://github.com/juangamella/gnies).

We use the following python packages for the other algorithms:

- `ges` for the [python implementation](https://github.com/juangamella/ges) of the GES algorithm
- `gies` for the [python implementation](https://github.com/juangamella/gies) of the GIES algorithm
- `causaldag` for UT-IGSP (see an [example](https://uhlerlab.github.io/causaldag/utigsp.html) and our [wrapper](https://github.com/juangamella/gnies-paper/blob/master/src/ut_igsp.py) including HSIC tests)

Additionally, we use the python package [`sempler`](https://github.com/juangamella/sempler) to generate synthetic and semi-synthetic data for our experiments.

## Installing dependencies

We ran our experiments using `python=3.9.9`, but they should work on any version above `3.8`. The Python dependencies live in [`requirements.txt`](requirements.txt). For your convenience, a makefile is included to create a python virtual environment and install the necessary Python dependencies. To do this, simply run

```sh
make venv
```

and then

```sh
source venv/bin/activate
```

to activate the virtual environment. Of course, you will need to be in a "make-capable" system (e.g. linux), and where you can invoke the python `venv` module. To run the notebooks from the virtual environment, create a new local kernel (while the environment is active):

```
ipython kernel install --user --name=.venv
```

and once inside the notebook select the kernel: `Kernel -> Change kernel -> .venv`.


## Reproducing the results

Below are the exact instructions to reproduce all the experiments and figures used in the paper. Please note that, without access to a HPC cluster, completion of the experiments may take days or weeks. We ran our experiments on the Euler cluster of ETH Zürich - see the files [`run_synthetic_experiments_cluster.sh`](run_synthetic_experiments_cluster.sh) and [`run_sachs_experiments_cluster.sh`](run_sachs_experiments_cluster.sh) for details (i.e. number of cores, expected completion time, etc).

We include all the datasets required to reproduce the experiments; the code to re-generate them can also be found in the files [`run_synthetic_experiments.sh`](run_synthetic_experiments.sh) and [`run_sachs_experiments.sh`](run_sachs_experiments.sh). Although not necessary, if you wish to re-generate the semi-synthetic datasets you will need some additional `R` dependencies (see [sempler's documentation](<TODO: Link)).


### Synthetic experiments (Figures 1 and 2)

1. Download and unpack the synthetic datasets
```bash
./download_synthetic_datasets.sh
```
2. Make sure the python environment is active (see above) and run the methods using the corresponding script. By default this will use a total of 4 threads (cores) to run the experiments; the number of threads can be set by editing the [script](run_synthetic_experiments.sh) and setting the variable `N_THREADS` to the desired value.
```bash
./run_synthetic_experiments.sh
```
3. The results are stored in the `synthetic_experiments/` under the sub-directory corresponding to each dataset.
4. To generate the figures, use notebooks [`figure_model_match.ipynb`](figure_model_match.ipynb) and [`figure_model_mismatch.ipynb`](figure_model_mismatch.ipynb). The resulting figures are stored in the `figures/` directory.

### Real and semi-synthetic data experiments with the Sachs dataset (Figure 3)

The procedure is similar to the synthetic experiments:

1. Download and unpack the Sachs dataset and the hybrid data
```bash
./download_sachs_datasets.sh
```
2. Make sure the python environment is active (see above) and run the methods using the corresponding script. By default this will use a total of 4 threads (cores) to run the experiments; the number of threads can be set by editing the [script](run_sachs_experiments.sh) and setting the variable `N_THREADS` to the desired value.
```bash
./run_sachs_experiments.sh
```
3. The results are stored in the `sach_experiments/` under the sub-directory corresponding to each dataset.
4. To generate the figures, use notebook [`figures_sach_experiments.ipynb`](figures_sachs_experiments.ipynb). The resulting figures are stored in the `figures/` directory.

## Repository structure

You will find the following/files directories:

- `src/`: contains the Python code to run the experiments. Each baseline is executed from its own python script:
  - `src/run_gnies.py` for GnIES
  - `src/run_utigsp.py` for UT-IGSP
  - `src/run_ges.py` for GES
  - `src/run_gies.py` for GIES
  - `src/run_sortnregress.py` for sortnregress
- `*_experiments` directories hold the datasets and the results from executing the experiments.
- `figure*.ipynb` are the jupyter notebooks used to generate the figures used in the paper. Figures are stored in the `figures/` directory.

## Feedback

If you need assistance or have feedback, you are more than welcome to write me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
