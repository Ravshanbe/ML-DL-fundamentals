# Household QR algorithm

This project provides a Python implementation of QR decomposition using the Householder reflection method. Given an m x n matrix A, it computes an m x m orthogonal matrix Q and an m x n upper-triangular matrix R such that A = QR.

## Testing

Several test cases are added in `pytest` (python testing framework). Just run pytest in the folder and it will run tests automatically. You can see tests in `test_qr.py`

## Requirements

*   Create virtual environment: python3 -m venv venv   -> source venv/bin/activate  (activate environment) -> pip install -r requirements.txt (install dependencies)
or
*   Python 3.x
*   NumPy (`pip install numpy`)
*   Pytest (for running tests) (`pip install pytest`)



## Usage

### Input File Format

The script reads the input matrix A from a text file. The file should be formatted as follows:

1.  The first line contains two integers, `m` and `n`, separated by whitespace, representing the dimensions (rows and columns) of the matrix A.
2.  The following `m` lines each contain `n` floating-point numbers, separated by whitespace, representing the rows of matrix A.


Exapmle:
3 2 -> dimension
1 2
3 4
5 6

### Running

Write following command in terminal in the folder of this file.
                                
`python3 qr_decomposition.py my_matrix.txt`  <-name of your matrix file
