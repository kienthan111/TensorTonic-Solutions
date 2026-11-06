import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x=np.array(x, dtype=np.float64)
    W1=np.array(W1, dtype=np.float64)
    W2=np.array(W2, dtype=np.float64)
    Ws=np.array(Ws, dtype=np.float64)
    y=np.dot(x, W1)
    y=np.maximum(y,0)
    y=np.dot(y, W2)
    y=y+np.dot(x, Ws)
    y=np.maximum(y,0)
    return y
    pass