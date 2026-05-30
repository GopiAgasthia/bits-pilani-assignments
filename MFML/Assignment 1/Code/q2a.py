import numpy as np


def lu_decomposition(A, verbose=True):
    """
    Perform LU decomposition on a square matrix.

    Decomposes matrix A into:
    - L: Lower triangular matrix with 1s on diagonal
    - U: Upper triangular matrix
    - Elementary matrices: The sequence of elimination matrices used

    Such that A = L * U

    Parameters:
        A (numpy.ndarray): Square matrix to decompose
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        tuple: (L, U, elementary_matrices)
            - L: Lower triangular matrix
            - U: Upper triangular matrix
            - elementary_matrices: List of elementary matrices used
    """
    # Convert to float for division operations
    A = A.astype(float)
    n = A.shape[0]

    # Initialize U as a copy of A
    U = A.copy()

    # Initialize L as identity matrix
    L = np.eye(n)

    # Track all elementary matrices used in the process
    elementary_matrices = []

    if verbose:
        print("\n=== LU Decomposition ===")
        print("\nInitial Matrix A:")
        print(A)
        print("\nInitial L (identity):")
        print(L)
        print("\nInitial U (copy of A):")
        print(U)

    # Process each column
    for col in range(n - 1):
        pivot = U[col][col]

        if verbose:
            print(f"\n--- Processing column {col} with pivot = {pivot:.3f} ---")

        # Eliminate all entries below the pivot
        for row in range(col + 1, n):
            # Calculate the multiplier (factor) needed to eliminate U[row][col]
            factor = U[row][col] / pivot

            # Store the factor in L matrix
            L[row][col] = factor

            # Create elementary matrix for this elimination step
            E = np.eye(n)
            E[row][col] = -factor
            elementary_matrices.append(E)

            if verbose:
                print(f"\nEliminating U[{row}][{col}]:")
                print(f"  Factor = U[{row}][{col}] / U[{col}][{col}] = {U[row][col]:.3f} / {pivot:.3f} = {factor:.3f}")
                print(f"  L[{row}][{col}] = {factor:.3f}")
                print(f"  R{row} = R{row} - ({factor:.3f}) * R{col}")

            # Perform row operation: row = row - factor * pivot_row
            for k in range(n):
                U[row][k] -= factor * U[col][k]

            if verbose:
                print(f"\nElementary Matrix E{len(elementary_matrices)}:")
                print(np.round(E, 3))
                print("\nCurrent U:")
                print(np.round(U, 3))
                print("\nCurrent L:")
                print(np.round(L, 3))

    if verbose:
        print("\n=== Final LU Decomposition ===")
        print("\nL (Lower triangular):")
        print(np.round(L, 3))
        print("\nU (Upper triangular):")
        print(np.round(U, 3))

    return L, U, elementary_matrices


def run_q2a(verbose=True):

    # Set seed for reproducibility
    np.random.seed(20)

    # Generate random dimension [1-5] for square matrix (LU requires square)
    n = np.random.randint(1, 6)

    # Generate random matrix and create symmetric positive definite matrix
    M = np.random.randint(1, 10, (n, n)).astype(float)
    A = np.dot(M.T, M)

    print("\nMatrix A:")
    print(A)

    L, U, elementary_matrices = \
        lu_decomposition(A, verbose)

    print("\nL:")
    print(L)

    print("\nU:")
    print(U)

    print("\nElementary Matrices:")

    for i, E in enumerate(elementary_matrices):
        print(f"\nE{i+1}")
        print(np.round(E, 3))

    print("\nVerification:")
    print(np.allclose(A, np.dot(L, U)))