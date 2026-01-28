import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    arr_x = np.array(x)
    return 1 / (1 + np.exp(-arr_x))