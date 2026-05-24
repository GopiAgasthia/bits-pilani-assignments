import numpy as np


def lu_decomposition(A, verbose=True):

    A = A.astype(float)

    n = A.shape[0]

    U = A.copy()

    L = np.eye(n)

    elementary_matrices = []

    if verbose:
        print("\n=== LU Decomposition ===")
        print("\nInitial Matrix A:")
        print(A)
        print("\nInitial L (identity):")
        print(L)
        print("\nInitial U (copy of A):")
        print(U)

    for col in range(n - 1):

        pivot = U[col][col]
        
        if verbose:
            print(f"\n--- Processing column {col} with pivot = {pivot:.3f} ---")

        for row in range(col + 1, n):

            factor = U[row][col] / pivot

            L[row][col] = factor

            E = np.eye(n)

            E[row][col] = -factor

            elementary_matrices.append(E)

            if verbose:
                print(f"\nEliminating U[{row}][{col}]:")
                print(f"  Factor = U[{row}][{col}] / U[{col}][{col}] = {U[row][col]:.3f} / {pivot:.3f} = {factor:.3f}")
                print(f"  L[{row}][{col}] = {factor:.3f}")
                print(f"  R{row} = R{row} - ({factor:.3f}) * R{col}")

            for k in range(n):
                U[row][k] -= factor * U[col][k]

            if verbose:
                print(f"\nElementary Matrix E{len(elementary_matrices)}:")
                print(E)
                print("\nCurrent U:")
                print(U)
                print("\nCurrent L:")
                print(L)

    if verbose:
        print("\n=== Final LU Decomposition ===")
        print("\nL (Lower triangular):")
        print(L)
        print("\nU (Upper triangular):")
        print(U)

    return L, U, elementary_matrices


def run_q2a():

    np.random.seed(20)

    M = np.random.randint(1, 10, (4, 4)).astype(float)

    A = np.dot(M.T, M)

    print("\nMatrix A:")
    print(A)

    L, U, elementary_matrices = \
        lu_decomposition(A)

    print("\nL:")
    print(L)

    print("\nU:")
    print(U)

    print("\nElementary Matrices:")

    for i, E in enumerate(elementary_matrices):
        print(f"\nE{i+1}")
        print(E)

    print("\nVerification:")
    print(np.allclose(A, np.dot(L, U)))