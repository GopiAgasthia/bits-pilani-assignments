import numpy as np

from utils import (
    rref,
    get_pivots_and_free_columns,
    particular_solution,
    null_space_basis
)


def run_q1b(verbose=True):

    # Set seed for reproducibility
    np.random.seed(15)

    # Generate random dimensions
    rows = np.random.randint(1, 6)
    cols = np.random.randint(rows, 6)  # Ensure cols >= rows for potential null space

    # Generate random matrix with values between -10 and 10
    A = np.random.randint(-10, 11, (rows, cols)).astype(float)

    R = rref(A, verbose)

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
        free_columns,
        verbose
    )

    print("\nParticular Solution:")
    print(particular)

    basis = null_space_basis(
        R,
        pivot_columns,
        free_columns,
        verbose
    )

    print("\nNull Space Basis:")

    for i, vec in enumerate(basis):
        print(f"v{i+1} = {vec}")

    # Print general solution
    print("\n=== General Solution ===")
    print("\nThe general solution is:")
    print("x = x_particular + c₁*v₁ + c₂*v₂ + ... + cₖ*vₖ")
    print("\nFor this system:")
    print(f"x_particular = {particular}")

    if len(basis) == 0:
        print("\nGeneral solution: x = x_particular (unique solution)")
    else:
        print("\nGeneral solution:")
        solution_str = f"x = {particular}"
        for i, vec in enumerate(basis):
            solution_str += f" + c{i+1}*{vec}"
        print(solution_str)

        print("\nIn component form:")
        for j in range(len(particular)):
            component_str = f"x{j+1} = {particular[j]:.1f}"
            for i, vec in enumerate(basis):
                if vec[j] != 0:
                    component_str += f" + {vec[j]:.1f}*c{i+1}" if vec[j] > 0 else f" - {abs(vec[j]):.1f}*c{i+1}"
            print(component_str)

        print(f"\nwhere c1, c2, ..., c{len(basis)} are free parameters (any real numbers)")