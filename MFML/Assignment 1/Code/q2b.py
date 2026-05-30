import numpy as np
import math


def cholesky_decomposition(A, verbose=True):
    """
    Perform Cholesky decomposition on a symmetric positive definite matrix.

    Decomposes matrix A into:
    - L: Lower triangular matrix

    Such that A = L * L^T

    The algorithm computes L column by column:
    - Diagonal elements: L[j][j] = sqrt(A[j][j] - sum(L[j][k]^2 for k < j))
    - Off-diagonal elements: L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]

    Parameters:
        A (numpy.ndarray): Symmetric positive definite matrix to decompose
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        numpy.ndarray: Lower triangular matrix L
    """
    n = A.shape[0]

    # Initialize L as zero matrix
    L = np.zeros((n, n))

    if verbose:
        print("\n=== Cholesky Decomposition ===")
        print("\nInitial Matrix A (must be symmetric positive definite):")
        print(A)
        print("\nInitial L (zeros):")
        print(L)

    # Process each row
    for i in range(n):
        # Process each column up to and including the diagonal
        for j in range(i + 1):

            # Case 1: Diagonal element L[j][j]
            if i == j:
                # Sum of squares of all previous elements in this row
                sum_value = 0
                for k in range(j):
                    sum_value += L[j][k] ** 2

                if verbose:
                    print(f"\n--- Computing diagonal element L[{j}][{j}] ---")
                    print(f"  Sum of L[{j}][k]^2 for k=0 to {j-1}: {sum_value:.6f}")
                    print(f"  L[{j}][{j}] = sqrt(A[{j}][{j}] - sum) = sqrt({A[j][j]:.6f} - {sum_value:.6f})")

                # Diagonal formula: sqrt(A[j][j] - sum of L[j][k]^2)
                L[j][j] = math.sqrt(A[j][j] - sum_value)

                if verbose:
                    print(f"  L[{j}][{j}] = {L[j][j]:.6f}")
                    print("\nCurrent L:")
                    print(np.round(L, 3))

            # Case 2: Off-diagonal element L[i][j] where i > j
            else:
                # Sum of products of corresponding elements in rows i and j
                sum_value = 0
                for k in range(j):
                    sum_value += L[i][k] * L[j][k]

                if verbose:
                    print(f"\n--- Computing off-diagonal element L[{i}][{j}] ---")
                    print(f"  Sum of L[{i}][k] * L[{j}][k] for k=0 to {j-1}: {sum_value:.6f}")
                    print(f"  L[{i}][{j}] = (A[{i}][{j}] - sum) / L[{j}][{j}]")
                    print(f"  L[{i}][{j}] = ({A[i][j]:.6f} - {sum_value:.6f}) / {L[j][j]:.6f}")

                # Off-diagonal formula: (A[i][j] - sum) / L[j][j]
                L[i][j] = (A[i][j] - sum_value) / L[j][j]

                if verbose:
                    print(f"  L[{i}][{j}] = {L[i][j]:.6f}")
                    print("\nCurrent L:")
                    print(np.round(L, 3))

    # Round to 3 decimal places to avoid floating point errors
    L = np.round(L, 3)

    if verbose:
        print("\n=== Final Cholesky Decomposition ===")
        print("\nL (Lower triangular):")
        print(L)
        print("\nVerification: L * L^T should equal A")

    return L


def run_q2b(verbose=True):

    # Set seed for reproducibility
    np.random.seed(25)

    # Generate random dimension [1-5] for square matrix (Cholesky requires square)
    n = np.random.randint(1, 6)

    # Generate random matrix and create symmetric positive definite matrix
    M = np.random.randint(1, 10, (n, n)).astype(float)
    A = np.dot(M.T, M)

    print("\nMatrix A:")
    print(A)

    L = cholesky_decomposition(A, verbose)

    print("\nL:")
    print(np.round(L, 3))

    print("\nVerification:")
    print(np.allclose(A, np.dot(L, L.T)))