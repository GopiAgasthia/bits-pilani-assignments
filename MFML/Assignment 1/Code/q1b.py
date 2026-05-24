import numpy as np

from utils import (
    rref,
    get_pivots_and_free_columns,
    particular_solution,
    null_space_basis
)


def run_q1b():

    A = np.array([
        [1, 2, -1, 3],
        [2, 4, 1, 8],
        [1, 1, 2, 2]
    ], dtype=float)

    R = rref(A)

    print("\nRREF:")
    print(R)

    pivot_columns, free_columns = \
        get_pivots_and_free_columns(R)

    print("\nPivot Columns:")
    print(pivot_columns)

    print("\nFree Columns:")
    print(free_columns)

    particular = particular_solution(
        R,
        pivot_columns,
        free_columns
    )

    print("\nParticular Solution:")
    print(particular)

    basis = null_space_basis(
        R,
        pivot_columns,
        free_columns
    )

    print("\nNull Space Basis:")

    for vec in basis:
        print(vec)