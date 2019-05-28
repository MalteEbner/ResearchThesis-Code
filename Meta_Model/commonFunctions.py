import numpy as np
import random

def softmax(x):
    x = np.array(x)
    y = np.exp(x)/sum(np.exp(x))
    z=y
    #z = np.floor(y*(2**30))#ensure softmax sums up to 1
    #z = z/(2**30)
    z[0] = 1-sum(z[1:])
    return z

def probsFromLosses(losses,exploitationFactor=10):
    # exploitationFactor ==0 equals all probs are the same
    probs = softmax(np.array(losses) / float(min(losses)) * -1*exploitationFactor)
    return probs

def pertRV(low,peak,high,g=4):
    a, b, c = [float(x) for x in [low, peak, high]]
    assert a<=b<=c, 'PERT "peak" must be greater than "low" and less than "high"'
    assert g>=0, 'PERT "g" must be non-negative'
    mu = (a + g*b + c)/(g + 2)
    if mu==b:
        a1 = a2 = 3.0
    else:
        a1 = ((mu - a)*(2*b - a - c))/((b - mu)*(c - a))
        a2 = a1*(c - mu)/(mu - a)

    rv = np.random.beta(a1,a2) #distributed in (0,1)
    rv = rv*(c-a)+a #distributed in (a,c)
    return rv

def chrashingCostIncrease(timeFactor):
    if timeFactor >=1:
        return 1
    else:
        costFactor = 560.7 * np.exp(-9.43*timeFactor)+0.955