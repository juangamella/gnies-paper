# To install GnIES:
# pip install gnies==0.2.0
import numpy as np
import pandas as pd
import gnies
import time

# --------------------------------------------------------------------
# Auxiliary Functions


# --------------------------------------------------------------------
# Execution parameters

seed = 42
runs = 1
lambdas = [0.01, 0.25]  # , 0.5, 0.75, 1]
splits = [0.5, 0.5]  # ratios for splits of the data, must sum to 1, e.g. [0.2,0.2,0.6]

# Gnies parameters
gnies_approach = "greedy"  # 'rank'
gnies_phases = ["backward"]  # ["forward", "backward"]  # For the GnIES outer procedure
gnies_verbosity = 0  # Set to 1 or 2 for different levels of GnIES verbosity

DATA_PATH = "./sachs/sachs_2005/"  # Path to the data files, relative to where the script is being run from
filenames = [
    "1. cd3cd28.csv",
    "2. cd3cd28icam2.csv",
    "3. cd3cd28+aktinhib.csv",
    "4. cd3cd28+g0076.csv",
    "5. cd3cd28+psitect.csv",
    "6. cd3cd28+u0126.csv",
    "7. cd3cd28+ly.csv",
    "8. pma.csv",
    "9. b2camp.csv",
]

# --------------------------------------------------------------------
# Load data
dataframes = [pd.read_csv(DATA_PATH + f) for f in filenames]
data = [df.to_numpy()[:, 0:5] for df in dataframes]

print("Loaded data")
N = 0
for filename, sample in zip(filenames, data):
    n, p = sample.shape
    N += n
    print('  "%s": %d observations of %d variables' % (filename, n, p))
print("Total: %d observations" % N)

# --------------------------------------------------------------------
# Run GnIES

# Set the inital set of intervention targets (if starting with the backward phase, start with full I)
p = data[0].shape[1]
I0 = set(range(p)) if gnies_phases[0] == "backward" else set()

# Arrays to store results. The dimensions (axes) are:
#   0 - index of the lambda
#   1 - index of the runs
#   2 - index of the fold
estimates = np.empty((len(lambdas), runs, len(splits), p, p), dtype=float)
I_estimates = np.empty((len(lambdas), runs, len(splits)), dtype=object)

for i, l in enumerate(lambdas):
    # Compute lambda
    lmbda = l * np.log(N)
    print("\nRunning for lambda = %0.4f * log(%d) = %0.2f" % (l, N, lmbda))
    for j in range(runs):
        print("  run %d - splitting into %d folds" % (j, len(splits)))
        # Split the data
        folds = gnies.utils.split_data(data, splits, random_state=seed + j)
        for k, fold in enumerate(folds):
            print("    running on fold %d/%d" % (k + 1, len(splits)))
            start = time.time()
            _score, icpdag, I = gnies.fit(
                fold, approach=gnies_approach, I0=I0, phases=gnies_phases, ges_lambda=lmbda, debug=gnies_verbosity
            )
            estimates[i, j, k, :, :] = icpdag
            I_estimates[i, j, k] = I
            print("      done in %0.2f seconds" % (time.time() - start))

print(estimates)
print(I_estimates)
