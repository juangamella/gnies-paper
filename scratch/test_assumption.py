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

import sempler
import sempler.utils as utils
import gnies.utils
import sempler.generators
import numpy as np
import sympy
import itertools

# --------------------------------------------------------------------
# Auxiliary functions to create and manipulate symbolic models, and to
# enumerate I-MAPs


def create_matrix(symbol, A):
    """Return a symbolic matrix with the sparsity patter of A, and
    elements represented by the given symbol."""
    rows, cols = A.shape
    return sympy.Matrix(rows, cols, lambda i, j: 0 if A[i, j] == 0 else sympy.var("%s_%d%d" % (symbol, i, j)))


def setup(A):
    """Given a DAG adjacency A, construct a symbolic connectivity matrix,
    noise-term variances and resulting covariance. Note: A is a graph
    adjacency where A[i,j] != 0 => j -> i.

    """
    p = len(A)
    B = create_matrix("B", A)
    omegas = [sympy.Symbol("w_%d" % i, positive=True) for i in range(p)]
    Omegas = sympy.Matrix.diag(omegas)
    # Omegas = sympy.Matrix.diag(['w_%d' % i for i in range(p)])
    I = np.eye(p)
    I_B = sympy.Matrix(I - B)
    Sigma = I_B.inv() @ Omegas @ I_B.inv().T
    Sigma.simplify()
    return B, Omegas, Sigma, I_B


def expand(b, pa, p):
    """Given a k-dimensional vector with k < p, return a p-dimensional
    vector with the elements of b at positions pa and zeros elsewhere"""
    vector = []
    count = 0
    for i in range(p):
        if i in pa:
            vector.append(b[count])
            count += 1
        else:
            vector.append(0)
    return sympy.Matrix([vector])


def regress(i, pa, Sigma):
    """Given a symbolic covariance matrix, find the coefficients from
    regressing i on pa"""
    pa = list(pa)
    b = Sigma[i, pa] @ Sigma[pa, pa].inv()
    b.simplify()
    return b


def fit_graph(Sigma, A):
    """
    Given a covariance matrix and a graph, find the expression for the
    weights using the regression coefficients of the conditional normal
    distributions.
    """
    p = len(A)
    B = []
    for i in range(len(A)):
        pa = sorted(list(utils.pa(i, A.T)))
        if pa != []:
            b = regress(i, pa, Sigma)
            row = expand(b, pa, p)
            B.append(row)
        else:
            B.append(sympy.Matrix.zeros(1, p))
    return sympy.Matrix(B)


def all_imaps(A, thresh=1e-15):
    """
    Enumerate all I-MAPs of the dag A using cholesky decompositions.
    """
    rng = np.random.default_rng(42)
    # First compute some random covariance matrix from the graph
    B = A * rng.uniform(0.5, 1, size=A.shape)
    cov = sempler.LGANM(B, (0, 0), (1, 2)).sample(population=True).covariance
    p = len(A)
    # Obtain all I-MAPs via a cholesky decomposition on all possible orderings
    Bs = []
    orderings = itertools.permutations(list(range(p)), p)
    for i, ordering in enumerate(orderings):
        # print("%d/%d  " % ((i + 1), np.math.factorial(p)), end="\r")
        rev_ordering = np.argsort(ordering)
        perm_cov = cov[:, ordering][ordering, :]
        # Cholesky decomposition on the permuted covariance matrix
        L = np.linalg.cholesky(perm_cov)
        B_perm = (np.eye(p) - np.linalg.inv(L / np.diag(L))).T
        B = B_perm[:, rev_ordering][rev_ordering, :]
        # Store resulting connectivity
        Bs.append(B)
        # print(ordering, utils.topological_ordering(B))
        # assert list(ordering) == utils.topological_ordering(B)
    Bs = np.array(Bs)
    imaps = (abs(Bs) > thresh).astype(int)
    imaps = np.unique(imaps, axis=0)
    return imaps


# --------------------------------------------------------------------
# Run experiments

seed = 1
p = 6  # number of variables
k = 1.7  # average density of Erdos-Renyi graphs
NUM_GRAPHS = 100  # total number of graphs


for i in range(NUM_GRAPHS):
    print("--------------------------------------------------")
    print("GRAPH %d/%d" % ((i + 1), NUM_GRAPHS))
    # Generate a symbolic model based on random graph A
    A = sempler.generators.dag_avg_deg(p, k, 1, 1, random_state=seed + i)
    B, Omegas, Sigma, I_B = setup(A.T)
    I_B_inv = I_B.inv()
    print()
    print("Enumerating MEC")
    # imaps = all_imaps(A)
    imaps = gnies.utils.mec(A)
    print("  done (%d I-MAPs)." % len(imaps))
    print()
    print("Checking hypothesis: assumption 2 cannot be violated")
    for j, imap in enumerate(imaps):
        print("  I-MAP %d/%d..." % (j + 1, len(imaps)))
        # Obtain expression for the imap's connectivity
        Bi = fit_graph(Sigma, imap.T)
        print("    fitted graph")
        I_Bi = sympy.Matrix(np.eye(p) - Bi)
        # Obtain expression for M matrix
        M = I_Bi @ I_B_inv
        print("    computed M matrix")
        M.simplify()
        # One by one, check if the diagonal elements of M can be set
        # to zero with some value of the parameters, i.e. weights B and
        # noise-term variances Omegas
        solutions = []
        for k in range(p):
            sol = sympy.solve(M[k, k])
            print(sol)
            assert sol == []
        print("    done.")
