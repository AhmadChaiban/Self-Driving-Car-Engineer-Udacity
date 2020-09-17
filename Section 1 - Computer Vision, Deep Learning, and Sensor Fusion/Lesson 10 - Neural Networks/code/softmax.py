import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

def softmax(L):
    exp_sum = sum(np.exp(L))
    softmax_results = []
    for score in L:
        softmax_results.append(np.exp(score)/exp_sum)
    return softmax_results

