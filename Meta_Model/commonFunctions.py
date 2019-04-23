import numpy as np

def softmax(x):
    x = np.array(x)
    y = np.exp(x)/sum(np.exp(x))
    z = np.floor(y*(2**20))#ensure softmax sums up to 1
    z = z/(2**20)
    z[0] = 1-sum(z[1:])
    return z

def probsFromLosses(losses,exploitationFactor=10):
    # exploitationFactor ==0 equals all probs are the same
    probs = softmax(np.array(losses) / float(min(losses)) * -1*exploitationFactor)
    return probs
