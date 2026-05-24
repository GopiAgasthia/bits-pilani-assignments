import numpy as np


def qr_decomposition(A, verbose=True):

    m, n = A.shape

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

    for j in range(n):

        if verbose:
            print(f"\n--- Processing column {j} ---")
            print(f"Starting with v = A[:, {j}]:")
            print(A[:, j])

        v = A[:, j].copy()

        for i in range(j):

            R[i][j] = np.dot(Q[:, i], A[:, j])

            if verbose:
                print(f"\nOrthogonalizing against Q[:, {i}]:")
                print(f"  R[{i}][{j}] = Q[:, {i}] · A[:, {j}] = {R[i][j]:.6f}")
                print(f"  v = v - {R[i][j]:.6f} * Q[:, {i}]")

            v = v - R[i][j] * Q[:, i]

            if verbose:
                print(f"  Updated v:")
                print(f"  {v}")

        R[j][j] = np.sqrt(np.dot(v, v))

        if verbose:
            print(f"\nNormalizing:")
            print(f"  R[{j}][{j}] = ||v|| = sqrt(v · v) = {R[j][j]:.6f}")
            print(f"  Q[:, {j}] = v / R[{j}][{j}]")

        Q[:, j] = v / R[j][j]

        if verbose:
            print(f"\nQ[:, {j}] (normalized):")
            print(Q[:, j])
            print("\nCurrent Q:")
            print(Q)
            print("\nCurrent R:")
            print(R)

    if verbose:
        print("\n=== Final QR Decomposition ===")
        print("\nQ (Orthonormal columns):")
        print(Q)
        print("\nR (Upper triangular):")
        print(R)
        print("\nVerification: Q * R should equal A")

    return Q, R


def run_q2c():

    np.random.seed(30)

    A = np.random.randint(1, 10, (5, 3)).astype(float)

    print("\nMatrix A:")
    print(A)

    Q, R = qr_decomposition(A)

    print("\nQ:")
    print(Q)

    print("\nR:")
    print(R)

    print("\nVerification:")
    print(np.allclose(A, np.dot(Q, R)))