import numpy as np


def ref(matrix, verbose=True):
    """Compute the row echelon form of a matrix.

    Parameters:
        matrix (numpy.ndarray): Input matrix.
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        numpy.ndarray: Matrix in row echelon form.
    """
    # Create a working copy as float to allow division
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0  # Track which row we're currently working on

    if verbose:
        print("\n=== Row Echelon Form (REF) ===")
        print("\nInitial Matrix:")
        print(A)

    # Process each column to find and use pivots
    for col in range(cols):
        # Step 1: Find a non-zero pivot in the current column
        pivot = None
        for r in range(pivot_row, rows):
            if A[r][col] != 0:
                pivot = r
                break

        # If no pivot found, skip this column
        if pivot is None:
            if verbose:
                print(f"\nColumn {col}: No pivot found, skipping")
            continue

        # Step 2: Swap rows to move pivot to the correct position
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
            if verbose:
                print(f"\nStep: Swap row {pivot} with row {pivot_row}")
                print(A)

        # Step 3: Eliminate all entries below the pivot
        if verbose:
            print(f"\nEliminating below pivot at position ({pivot_row}, {col})")

        for r in range(pivot_row + 1, rows):
            # Calculate the factor needed to eliminate A[r][col]
            factor = A[r][col] / A[pivot_row][col]

            if verbose and factor != 0:
                print(f"  R{r} = R{r} - ({factor:.3f}) * R{pivot_row}")

            # Subtract factor * pivot_row from current row
            for c in range(col, cols):
                A[r][c] -= factor * A[pivot_row][c]

        if verbose:
            print("Matrix after elimination:")
            print(A)

        # Move to the next pivot row
        pivot_row += 1
        if pivot_row == rows:
            break

    # Round to 3 decimal places to avoid floating point errors
    result = np.round(A, 3)
    if verbose:
        print("\n=== Final REF Matrix ===")
        print(result)

    return result


def rref(matrix, verbose=True):

    """Compute the reduced row echelon form of a matrix.

    Parameters:
        matrix (numpy.ndarray): Input matrix.
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        numpy.ndarray: Matrix in reduced row echelon form.
    """

    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0  # Track which row we're currently working on

    if verbose:
        print("\n=== Reduced Row Echelon Form (RREF) ===")
        print("\nInitial Matrix:")
        print(A)

    # Process each column to find and use pivots
    for col in range(cols):
        # Step 1: Find a non-zero pivot in the current column
        pivot = None
        for r in range(pivot_row, rows):
            if A[r][col] != 0:
                pivot = r
                break

        # If no pivot found, skip this column
        if pivot is None:
            if verbose:
                print(f"\nColumn {col}: No pivot found, skipping")
            continue

        # Step 2: Swap rows to move pivot to the correct position
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
            if verbose:
                print(f"\nStep: Swap row {pivot} with row {pivot_row}")
                print(A)

        # Step 3: Normalize the pivot row (make pivot = 1)
        pivot_value = A[pivot_row][col]

        if verbose:
            print(f"\nNormalizing row {pivot_row} (dividing by {pivot_value:.3f})")

        # Divide entire row by the pivot value
        for c in range(cols):
            A[pivot_row][c] /= pivot_value

        if verbose:
            print("Matrix after normalization:")
            print(A)

        # Step 4: Eliminate ALL other entries in the pivot column (not just below)
        if verbose:
            print(f"\nEliminating all entries in column {col} except pivot")

        for r in range(rows):
            # Skip the pivot row itself
            if r != pivot_row:
                # Calculate the factor needed to eliminate A[r][col]
                factor = A[r][col]

                if verbose and factor != 0:
                    print(f"  R{r} = R{r} - ({factor:.3f}) * R{pivot_row}")

                # Subtract factor * pivot_row from current row
                for c in range(cols):
                    A[r][c] -= factor * A[pivot_row][c]

        if verbose:
            print("Matrix after elimination:")
            print(A)

        # Move to the next pivot row
        pivot_row += 1
        if pivot_row == rows:
            break

    # Round to 3 decimal places to avoid floating point errors
    result = np.round(A, 3)
    if verbose:
        print("\n=== Final RREF Matrix ===")
        print(result)

    return result


def get_pivots_and_free_columns(rref_matrix):

    """Identify pivot and free columns from a reduced row echelon matrix.

    Pivot columns correspond to leading variables, while free columns
    correspond to variables that can be chosen freely.

    Parameters:
        rref_matrix (numpy.ndarray): Matrix in reduced row echelon form.

    Returns:
        tuple[list[int], list[int]]: A tuple of pivot columns and free columns.
    """

    rows, cols = rref_matrix.shape
    pivot_columns = []

    for r in range(rows):
        for c in range(cols):
            if abs(rref_matrix[r][c] - 1) < 1e-10:
                is_pivot = True
                for k in range(rows):
                    if k != r and abs(rref_matrix[k][c]) > 1e-10:
                        is_pivot = False
                        break
                if is_pivot:
                    pivot_columns.append(c)
                    break

    free_columns = [c for c in range(cols - 1) if c not in pivot_columns]
    return pivot_columns, free_columns


def particular_solution(rref_matrix, pivot_columns, free_columns, verbose=True):

    """Compute a particular solution for the augmented system in RREF.

    Free variables are set to zero, and pivot variables are assigned the
    constant term from each pivot row.

    Parameters:
        rref_matrix (numpy.ndarray): Augmented matrix in reduced row echelon form.
        pivot_columns (list[int]): Indices of pivot columns.
        free_columns (list[int]): Indices of free variable columns.
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        numpy.ndarray: Particular solution vector.
    """

    rows, cols = rref_matrix.shape
    variables = cols - 1
    x = np.zeros(variables)

    if verbose:
        print("\n=== Computing Particular Solution ===")
        print("\nStrategy: Set all free variables to 0, solve for pivot variables")
        print(f"Free variables (set to 0): {free_columns}")
        print(f"Pivot variables (solve from RREF): {pivot_columns}")
        print("\nInitial solution vector (all zeros):")
        print(x)

    for r in range(rows):
        pivot_col = None
        for c in pivot_columns:
            if abs(rref_matrix[r][c] - 1) < 1e-10:
                pivot_col = c
                break
        if pivot_col is not None:
            if verbose:
                print(f"\nRow {r}: Pivot at column {pivot_col}")
                print(f"  Setting x[{pivot_col}] = {rref_matrix[r][-1]:.1f} (constant term)")
            x[pivot_col] = rref_matrix[r][-1]
            if verbose:
                print(f"  Current solution: {x}")

    if verbose:
        print("\n=== Final Particular Solution ===")
        print(x)

    return x


def null_space_basis(rref_matrix, pivot_columns, free_columns, verbose=True):

    """Construct a basis for the null space of the coefficient matrix.

    Each free variable is set to 1 in turn, and pivot variables are computed
    to satisfy the homogeneous system A x = 0.

    Parameters:
        rref_matrix (numpy.ndarray): Augmented matrix in reduced row echelon form.
        pivot_columns (list[int]): Indices of pivot columns.
        free_columns (list[int]): Indices of free variable columns.
        verbose (bool): If True, print intermediate steps. Default is True.

    Returns:
        list[numpy.ndarray]: Basis vectors for the null space.
    """

    rows, cols = rref_matrix.shape
    variables = cols - 1
    basis = []

    if verbose:
        print("\n=== Computing Null Space Basis ===")
        print("\nStrategy: Set each free variable to 1 (one at a time), solve for pivot variables")
        print(f"Number of free variables: {len(free_columns)}")
        print(f"Free variable columns: {free_columns}")
        print(f"Pivot variable columns: {pivot_columns}")

    for idx, free_col in enumerate(free_columns):
        if verbose:
            print(f"\n--- Basis Vector {idx + 1} (for free variable x{free_col + 1}) ---")

        vector = np.zeros(variables)
        vector[free_col] = 1

        if verbose:
            print(f"Set x[{free_col}] = 1 (free variable)")
            print(f"Initial vector: {vector}")

        for r in range(rows):
            pivot_col = None
            for c in pivot_columns:
                if abs(rref_matrix[r][c] - 1) < 1e-10:
                    pivot_col = c
                    break
            if pivot_col is not None:
                # Express the pivot variable in terms of the current free variable.
                coeff = -rref_matrix[r][free_col]
                if verbose:
                    print(f"\nRow {r}: Pivot at column {pivot_col}")
                    print(f"  From RREF: x[{pivot_col}] + ({rref_matrix[r][free_col]:.1f})*x[{free_col}] = 0")
                    print(f"  Therefore: x[{pivot_col}] = {coeff:.1f}")
                vector[pivot_col] = coeff
                if verbose:
                    print(f"  Current vector: {vector}")

        basis.append(vector)

        if verbose:
            print(f"\nBasis vector {idx + 1}: {vector}")

    if verbose:
        print("\n=== Final Null Space Basis ===")
        for idx, vec in enumerate(basis):
            print(f"v{idx + 1} = {vec}")

    return basis