import numpy as np
from utils import ref, rref


def run_q1a(verbose = True):

    # Set seed for reproducibility
    np.random.seed(10)

    # Generate random dimensions
    rows = np.random.randint(1, 6)
    cols = np.random.randint(rows, 6)  # Ensure cols >= rows

    # Generate random matrix with values between -10 and 10
    A = np.random.randint(-10, 11, (rows, cols)).astype(float)

    print("\nOriginal Matrix:")
    print(A)

    print("\nREF:")
    print(ref(A, verbose))

    print("\nRREF:")
    print(rref(A, verbose))