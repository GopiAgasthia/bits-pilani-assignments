import numpy as np
from utils import ref, rref


def run_q1a():

    A = np.array([
        [1, 2, -1, 3],
        [2, 4, 1, 8],
        [1, 1, 2, 2]
    ], dtype=float)

    print("\nOriginal Matrix:")
    print(A)

    print("\nREF:")
    print(ref(A))

    print("\nRREF:")
    print(rref(A))