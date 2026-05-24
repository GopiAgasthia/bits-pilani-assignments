import numpy as np
import math


def cholesky_decomposition(A, verbose=True):

    n = A.shape[0]

    L = np.zeros((n, n))

    if verbose:
        print("\n=== Cholesky Decomposition ===")
        print("\nInitial Matrix A (must be symmetric positive definite):")
        print(A)
        print("\nInitial L (zeros):")
        print(L)

    for i in range(n):

        for j in range(i + 1):

            if i == j:

                sum_value = 0

                for k in range(j):
                    sum_value += L[j][k] ** 2

                if verbose:
                    print(f"\n--- Computing diagonal element L[{j}][{j}] ---")
                    print(f"  Sum of L[{j}][k]^2 for k=0 to {j-1}: {sum_value:.6f}")
                    print(f"  L[{j}][{j}] = sqrt(A[{j}][{j}] - sum) = sqrt({A[j][j]:.6f} - {sum_value:.6f})")

                L[j][j] = math.sqrt(
                    A[j][j] - sum_value
                )

                if verbose:
                    print(f"  L[{j}][{j}] = {L[j][j]:.6f}")
                    print("\nCurrent L:")
                    print(L)

            else:

                sum_value = 0

                for k in range(j):
                    sum_value += L[i][k] * L[j][k]

                if verbose:
                    print(f"\n--- Computing off-diagonal element L[{i}][{j}] ---")
                    print(f"  Sum of L[{i}][k] * L[{j}][k] for k=0 to {j-1}: {sum_value:.6f}")
                    print(f"  L[{i}][{j}] = (A[{i}][{j}] - sum) / L[{j}][{j}]")
                    print(f"  L[{i}][{j}] = ({A[i][j]:.6f} - {sum_value:.6f}) / {L[j][j]:.6f}")

                L[i][j] = (
                    A[i][j] - sum_value
                ) / L[j][j]

                if verbose:
                    print(f"  L[{i}][{j}] = {L[i][j]:.6f}")
                    print("\nCurrent L:")
                    print(L)

    if verbose:
        print("\n=== Final Cholesky Decomposition ===")
        print("\nL (Lower triangular):")
        print(L)
        print("\nVerification: L * L^T should equal A")

    return L


def run_q2b():

    np.random.seed(20)

    M = np.random.randint(1, 10, (4, 4)).astype(float)

    A = np.dot(M.T, M)

    print("\nMatrix A:")
    print(A)

    L = cholesky_decomposition(A)

    print("\nL:")
    print(L)

    print("\nVerification:")
    print(np.allclose(A, np.dot(L, L.T)))