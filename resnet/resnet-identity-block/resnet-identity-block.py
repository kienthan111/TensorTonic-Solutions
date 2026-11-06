import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x=np.array(x, dtype=np.float64)
    W1=np.array(W1, dtype=np.float64)
    W2=np.array(W2, dtype=np.float64)
    y=np.dot(x, W1.T)
    y=np.maximum(y, 0)
    y=np.dot(y,W2.T)
    y=y+x
    y=np.maximum(y,0)
    return y
    pass