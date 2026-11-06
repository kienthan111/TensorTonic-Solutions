import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x=np.array(x, dtype=np.float64)
    W1=np.array(W1, dtype=np.float64)
    W2=np.array(W2, dtype=np.float64)
    W3=np.array(W3, dtype=np.float64)
    if Ws is None:
        Ws=x
    else:
        Ws=np.array(Ws, dtype=np.float64)
        Ws=np.dot(x,Ws)
    y=np.dot(x, W1)
    y=np.maximum(y,0)
    y=np.dot(y, W2)
    y=np.maximum(y, 0)
    y=np.dot(y, W3)
    y=y+Ws
    y=np.maximum(y,0)
    return np.round(y,4).tolist()
    pass