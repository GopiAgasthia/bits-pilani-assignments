import numpy as np

from utils import (
    ref,
    rref,
    get_pivots_and_free_columns,
    particular_solution,
    null_space_basis
)


def run_q1c():

    np.random.seed(10)

    A = np.random.randint(-9, 10, (6, 9)).astype(float)

    b = np.random.randint(-9, 10, (6, 1)).astype(float)

    Aug = np.hstack((A, b))

    print("\nMatrix A:")
    print(A)

    print("\nVector b:")
    print(b)

    REF = ref(Aug)

    print("\nREF:")
    print(REF)

    RREF = rref(Aug)

    print("\nRREF:")
    print(RREF)

    pivot_columns, free_columns = \
        get_pivots_and_free_columns(RREF)

    print("\nPivot Columns:")
    print(pivot_columns)

    print("\nFree Columns:")
    print(free_columns)

    particular = particular_solution(
        RREF,
        pivot_columns,
        free_columns
    )

    print("\nParticular Solution:")
    print(particular)

    basis = null_space_basis(
        RREF,
        pivot_columns,
        free_columns
    )

    print("\nNull Space Basis:")

    for vec in basis:
        print(vec)