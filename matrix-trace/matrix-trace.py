import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    s=0
    A=np.array(A)
    for i in range(len(A)):
        s=s+A[i,i]
    return s
    pass
