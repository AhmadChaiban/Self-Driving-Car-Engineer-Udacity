import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    cross_entropy = 0
    for i in range(len(P)):
        cross_entropy += Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i])
    return -1 * cross_entropy


### Solution

import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))