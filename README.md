# Experiment Repository for <TODO: paper title>

This repository contains the code to reproduce the experiments and figures for the paper *"<TODO: paper title>"*, by JL. Gamella, A. Taeb, C. Heinze-Deml and P. Bühlmann. This README is not intended to be completely self-explanatory, and should be read alongside the manuscript (<TODO: link to arxiv>).

## Software package

If you're interested in using the GnIES algorithm described in the paper for your own work, it is available as a separate and well-documented python package called `gnies`. You can find more information on its own repository at [github.com/juangamella/gnies](https://github.com/juangamella/gnies).

This repository contains only the code to reproduce the results from the paper.

## Installing Dependencies

We ran our experiments using `python=<TODO: version>` and `R=<TODO: version>`. The required R packages can be found in [`R_requirements.txt`](R_requirements.txt). The Python dependencies live in [`requirements.txt`](requirements.txt).

For your convenience, a makefile is included to create a python virtual environment and install the necessary Python dependencies. To do this, simply run

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

Below are the exact instructions to reproduce all the experiments and figures used in the paper. Please note that, without access to a HPC cluster, completion of the experiments may take days or weeks. We ran our experiments on the Euler cluster of ETH Zürich - see the files [`run_baselines_cluster.sh`](run_baselines_cluster.sh) and [`run_comparisons_cluster.sh`](run_comparisons_cluster.sh) for details (i.e. number of cores, expected completion time, etc).

### Synthetic Experiments (figure <TODO: Figure>)

1. Execute the script [`run_baselines.sh`](run_baselines.sh). It will use a total of 4 threads (cores) to run the experiments; the number of threads can be set by editing the script and setting the variable `N_THREADS` to the desired value.
2. The results are stored in the `baseline_experiments/` directory.
3. To generate the figures, use notebooks [`figures_baseline_2.ipynb`](figures_baseline_2.ipynb) and [`figures_baseline_3.ipynb`](figures_baseline_3.ipynb), appropriately replacing the existing result filenames by those from step 2. The resulting figures are stored in the `figures/` directory.

### Real and Hybrid Data Experiments with the Sachs Dataset (figure <TODO: Figure>)

1. Execute the script [`run_comparisons.sh`](run_comparisons.sh). It will use a total of 4 threads (cores) to run the experiments; the number of threads can be set by editing the script and setting the variable `N_THREADS` to the desired value.
2. The results are stored in the `comparison_experiments/` directory.
3. To generate the figures, use notebook [`figures_comparisons.ipynb`](figures_comparisons.ipynb), appropriately replacing the existing result filenames by those from step 2. The resulting figures are stored in the `figures/` directory.

## Repository structure

You will find the following/files directories:

- `src/` contains the python and R code to run the experiments.
- `*_experiments` directories hold the results from executing the experiments.
- `figures_*.ipynb` are the jupyter notebooks used to generate the figures used in the paper. After execution, they are stored in the `figures/` directory.

## Feedback

If you need assistance or have feedback, you are more than welcome to write me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
