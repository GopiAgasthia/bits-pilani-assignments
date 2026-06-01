import numpy as np


def qr_decomposition(A, verbose=True):
    """
    Perform QR decomposition using the Gram-Schmidt orthogonalization process.

    Decomposes matrix A into:
    - Q: Matrix with orthonormal columns (Q^T * Q = I)
    - R: Upper triangular matrix

    Such that A = Q * R

    The Gram-Schmidt process:
    1. For each column of A, start with that column vector
    2. Subtract projections onto all previous orthonormal vectors
    3. Normalize the resulting vector to get the next column of Q
    4. Store the projection coefficients and norm in R

    Parameters:
        A (numpy.ndarray): Matrix to decompose (m x n)
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        tuple: (Q, R)
            - Q: Matrix with orthonormal columns (m x n)
            - R: Upper triangular matrix (n x n)
    """
    m, n = A.shape

    # Initialize Q and R as zero matrices
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    if verbose:
        print("\n=== QR Decomposition (Gram-Schmidt Process) ===")
        print("\nInitial Matrix A:")
        print(A)
        print(f"\nMatrix dimensions: {m} x {n}")
        print("\nInitial Q (zeros):")
        print(Q)
        print("\nInitial R (zeros):")
        print(R)

    # Process each column of A
    for j in range(n):
        if verbose:
            print(f"\n--- Processing column {j} ---")
            print(f"Starting with v = A[:, {j}]:")
            print(A[:, j])

        # Start with the j-th column of A
        v = A[:, j].copy()

        # Orthogonalize against all previous columns of Q
        for i in range(j):
            # Compute projection coefficient: R[i][j] = Q[:, i] · A[:, j]
            R[i][j] = np.dot(Q[:, i], A[:, j])

            if verbose:
                print(f"\nOrthogonalizing against Q[:, {i}]:")
                print(f"  R[{i}][{j}] = Q[:, {i}] · A[:, {j}] = {R[i][j]:.6f}")
                print(f"  v = v - {R[i][j]:.6f} * Q[:, {i}]")

            # Subtract the projection: v = v - R[i][j] * Q[:, i]
            v = v - R[i][j] * Q[:, i]

            if verbose:
                print(f"  Updated v:")
                print(f"  {v}")

        # Compute the norm of the orthogonalized vector
        R[j][j] = np.sqrt(np.dot(v, v))

        if verbose:
            print(f"\nNormalizing:")
            print(f"  R[{j}][{j}] = ||v|| = sqrt(v · v) = {R[j][j]:.6f}")
            print(f"  Q[:, {j}] = v / R[{j}][{j}]")

        # Normalize to get the j-th column of Q
        Q[:, j] = v / R[j][j]

        if verbose:
            print(f"\nQ[:, {j}] (normalized):")
            print(np.round(Q[:, j], 3))
            print("\nCurrent Q:")
            print(np.round(Q, 3))
            print("\nCurrent R:")
            print(np.round(R, 3))

    # Round to 3 decimal places to avoid floating point errors
    Q = np.round(Q, 3)
    R = np.round(R, 3)

    if verbose:
        print("\n=== Final QR Decomposition ===")
        print("\nQ (Orthonormal columns):")
        print(Q)
        print("\nR (Upper triangular):")
        print(R)
        print("\nVerification: Q * R should equal A")

    return Q, R


def run_q2c(verbose=True):

    # Set seed for reproducibility
    np.random.seed(30)

    # Generate random dimensions [1-5] x [1-5] with rows >= cols (tall or square)
    cols = np.random.randint(1, 6)
    rows = np.random.randint(cols, 6)  # Ensure rows >= cols for QR

    # Generate random matrix with values between 1 and 10
    A = np.random.randint(1, 10, (rows, cols)).astype(float)

    print("\nMatrix A:")
    print(A)

    Q, R = qr_decomposition(A, verbose)

    print("\nQ:")
    print(np.round(Q, 3))

    print("\nR:")
    print(np.round(R, 3))

    print("\nVerification:")
    print(np.allclose(A, np.dot(Q, R)))