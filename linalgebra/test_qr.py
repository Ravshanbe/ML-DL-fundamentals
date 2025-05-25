import numpy as np
import pytest
import os
from qr_decomposition import householder_qr, vector_norm



def check_properties(A, Q, R, tol=1e-10, matrix_name="Test Matrix"):
    m, n = A.shape
    
    # 1. A approx QR
    A_reconstructed = np.dot(Q, R)
    diff_A_QR_norm = vector_norm((A - A_reconstructed).flatten())
    norm_A = vector_norm(A.flatten())
    relative_error_A_QR = diff_A_QR_norm / (norm_A if norm_A > tol else 1.0)
    
    print(f"[{matrix_name}] ||A - QR||/||A||: {relative_error_A_QR:.2e}")
    assert relative_error_A_QR < tol or norm_A < tol, \
        f"[{matrix_name}] A - QR reconstruction error too large: {relative_error_A_QR:.2e}"
    
    # 2. Q is orthogonal: Q^T Q approx I
    if m > 0:
        QTQ = np.dot(Q.T, Q)
        I_m = np.eye(m)
        diff_QTQ_I_norm = vector_norm((QTQ - I_m).flatten())
        
        print(f"[{matrix_name}] ||Q^T Q - I||: {diff_QTQ_I_norm:.2e}")
        assert diff_QTQ_I_norm < tol, \
            f"[{matrix_name}] Q^T Q - I orthogonality error too large: {diff_QTQ_I_norm:.2e}"
    elif m == 0: # Q is 0x0, QTQ is 0x0, I_m is 0x0. This is fine.
        print(f"[{matrix_name}] Q is 0x0, orthogonality check skipped/trivial.")


    # 3. R is upper triangular (m x n)
    for i_row in range(m):
        for j_col in range(n):
            if i_row > j_col:
                assert abs(R[i_row, j_col]) < tol, \
                    f"[{matrix_name}] R[{i_row},{j_col}] = {R[i_row,j_col]:.2e} is not zero for upper triangular (tol {tol})"
    print(f"[{matrix_name}] R is upper triangular: Yes (within tolerance)")


# Test cases
test_matrices_data = [
    ([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], "A (3x3 Symmetric Positive Definite)"),
    ([[1., 3., 5., 7.], [2., 4., 6., 8.]], "B (2x4 Wide, Rank Deficient)"),
    ([[1., 2.], [3., 4.], [5., 6.]], "C (3x2 Tall)"),
    ([[1.]], "D (1x1 Scalar)"),
    ([[1., 2.]], "E (1x2 Row Vector)"),
    ([[1.], [2.]], "F (2x1 Column Vector)"),
    ([[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]], "G (Already Upper Triangular)"),
    ([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], "H (Permutation Matrix)"),
    ([[1e-12, 2.], [3., 4.]], "I (Small First Element)"),
    ([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], "J (Rank 1 Matrix)"),
    ([[6., 5., 0.], [5., 1., 4.], [0., 4., 3.]], "K (Wikipedia Example Symmetric)"),
    (np.random.rand(5, 5).tolist(), "Random 5x5"),
    (np.random.rand(5, 3).tolist(), "Random 5x3 Tall"),
    (np.random.rand(3, 5).tolist(), "Random 3x5 Wide"),
    (np.zeros((3,3)).tolist(), "Zero Matrix 3x3"),
    (np.zeros((2,4)).tolist(), "Zero Matrix 2x4"),
    (np.array([[1.0, 2.0, 3.0]]), "Row Vector 1x3 (explicit)"),
    (np.array([[1.0],[2.0],[3.0]]), "Column Vector 3x1 (explicit)"),
    (np.eye(4).tolist(), "Identity Matrix 4x4"),
    ([[1.,0.,0.],[0.,1e-15,0.],[0.,0.,1.]], "Near Singular / Small Diagonal in R"),
    (np.empty((3,0)).tolist(), "Empty Matrix (3x0)"),
]

@pytest.mark.parametrize("A_list, name", test_matrices_data)
def test_householder_qr_parametrized(A_list, name):
    print(f"\n--- Testing {name} ---")
    A = np.array(A_list, dtype=float)
    
    if A.shape[0] == 0 or A.shape[1] == 0:
        Q, R = householder_qr(A)
        assert Q.shape == (A.shape[0], A.shape[0]), f"Q shape incorrect for {name}"
        assert R.shape == A.shape, f"R shape incorrect for {name}"
        if A.size > 0 : 
             check_properties(A, Q, R, tol=1e-9, matrix_name=name)
        return

    Q, R = householder_qr(A)
    check_properties(A, Q, R, tol=1e-9, matrix_name=name)

