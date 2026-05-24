import numpy as np

from q2c import qr_decomposition


def run_q2d():

    np.random.seed(50)

    A = np.random.randint(1, 10, (7, 5)).astype(float)

    print("\nRandom 7x5 Matrix:")
    print(A)

    Q, R = qr_decomposition(A)

    print("\nQ:")
    print(Q)

    print("\nR:")
    print(R)

    print("\nDiagonal Entries of R:")

    for i in range(R.shape[0]):
        print(R[i][i])

    print("\nVerification:")
    print(np.allclose(A, np.dot(Q, R)))