import numpy as np

from q2c import qr_decomposition


def run_q2d(verbose=True):

    np.random.seed(50)

    A = np.random.randint(1, 10, (7, 5)).astype(float)

    print("\nRandom 7x5 Matrix:")
    print(A)

    Q, R = qr_decomposition(A, verbose)

    print("\nQ:")
    print(np.round(Q, 3))

    print("\nR:")
    print(np.round(R, 3))

    print("\nDiagonal Entries of R:")

    for i in range(R.shape[0]):
        print(np.round(R[i][i], 3))

    print("\nVerification:")
    print(np.allclose(A, np.dot(Q, R)))