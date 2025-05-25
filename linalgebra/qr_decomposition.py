import numpy as np
import sys

def safe_sign(x):

    if x >= 0:
        return 1.0
    else:
        return -1.0

def vector_norm(vec):
    return np.sqrt(np.dot(vec, vec))

def householder_qr(A_orig):
    m, n = A_orig.shape
    
    R = A_orig.copy().astype(float) 
    Q = np.eye(m, dtype=float) # identity matrix
    limit_k = min(n, m - 1 if m > 0 else 0)

    if m == 0: 
        return Q, R 

    for k in range(limit_k):
        x = R[k:m, k].copy() 
        
        norm_x = vector_norm(x)
        
        if norm_x < 1e-15: # Threshold for x being effectively zero
            continue 


        s = safe_sign(x[0])
        alpha = -s * norm_x

        u = x.copy()
        u[0] -= alpha 
        norm_u = vector_norm(u)

        if norm_u < 1e-15: # already zero case
            continue
        
        v = u / norm_u 


        v_T_R_sub = np.dot(v, R[k:m, k:n])
   
        R[k:m, k:n] -= 2 * np.outer(v, v_T_R_sub)
    
        Q_sub_v = np.dot(Q[:, k:m], v)
        Q[:, k:m] -= 2 * np.outer(Q_sub_v, v)
        
    for i_row in range(m):
        for j_col in range(n):
            if i_row > j_col and abs(R[i_row, j_col]) < 1e-12: 
                 R[i_row, j_col] = 0.0
                 
    return Q, R

def read_matrix_from_file(filename):
    with open(filename, "r") as f:
        m, n = map(int, f.readline().split())
        A = np.zeros((m, n))
        for i in range(m):
            row_str = f.readline().split()
            if len(row_str) != n:
                raise ValueError(f"Row {i} has {len(row_str)}, expected {n}")
            A[i, :] = list(map(float, row_str))
    return A

def save_matrix(matrix, filename):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(map(lambda x: f"{x:.15e}", row)) + "\n")

def show_matrix(matrix, name='A'):
    print(f"Matrix {name} ({matrix.shape[0]}x{matrix.shape[1]}):")
    if matrix.size == 0:
        print("[]")
        return
    
    for row in matrix:
        print(" [" + " ".join(f"{x:12.6e}" for x in row) + "]") 
    print("-" * (15 * matrix.shape[1] if matrix.shape[1] > 0 else 20))


def main():
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        print("Usage: python qr_decomposition.py <input_matrix_file>")
        print("Creating a default 'matrix.txt' for demonstration.")
        default_A_content = "3 3\n2 -1 0\n-1 2 -1\n0 -1 2"
        with open("matrix.txt", "w") as f:
            f.write(default_A_content)
        input_filename = "matrix.txt"
        print("Default 'matrix.txt' created with a 3x3 matrix.")



    A = read_matrix_from_file(input_filename)

    show_matrix(A, "A_input")

    Q, R = householder_qr(A)

    save_matrix(Q, "Q.txt")
    save_matrix(R, "R.txt")
    print(f"Output Q saved to Q.txt")
    print(f"Output R saved to R.txt")


    show_matrix(Q, "Q_computed")
    show_matrix(R, "R_computed")

    m, n = A.shape
    tol = 1e-10 
    
    if A.size > 0 : 
        A_reconstructed = np.dot(Q, R)
        diff_A_QR_norm = vector_norm((A - A_reconstructed).flatten())
        norm_A = vector_norm(A.flatten())
        relative_error_A_QR = diff_A_QR_norm / (norm_A if norm_A > tol else 1.0)
        print(f"Verification ||A - QR|| / ||A||: {relative_error_A_QR:.2e} (target < {tol:.1e})")
        if relative_error_A_QR > tol and norm_A > tol :
             print("Warning: ||A - QR|| / ||A|| is above tolerance.")

        
        if m > 0: 
            QTQ = np.dot(Q.T, Q)
            I_m = np.eye(m)
            diff_QTQ_I_norm = vector_norm((QTQ - I_m).flatten())
            print(f"Verification ||Q^T Q - I||: {diff_QTQ_I_norm:.2e} (target < {tol:.1e})")
            if diff_QTQ_I_norm > tol :
                 print("Warning: ||Q^T Q - I|| is above tolerance.")

        
        is_upper_triangular = True
        for i_row in range(m):
            for j_col in range(n):
                if i_row > j_col and abs(R[i_row, j_col]) > tol:
                    is_upper_triangular = False
                    print(f"Warning: R[{i_row},{j_col}] = {R[i_row,j_col]:.2e} is not zero for upper triangular (tol {tol:.1e})")
                    break
            if not is_upper_triangular:
                break
        if is_upper_triangular:
            print("R is upper triangular? Yes (within tolerance)")
        else:
            print("R is upper triangular? No (failed tolerance check)")

if __name__ == "__main__":
    main()