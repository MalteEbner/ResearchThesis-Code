import numpy as np
import random

def softmax(x):
    x = np.array(x)
    y = np.exp(x)/sum(np.exp(x))
    z=y
    z[0] = 1-sum(z[1:]) # ensure softmax sums up to 1
    return z

def probsFromLosses(losses,exploitationFactor=10):
    # exploitationFactor ==0 equals all probs are the same
    losses = np.array(losses)

    #scale losses to [0 , 1]
    min_ = float(min(losses))
    max_ = float(max(losses))
    losses -= min_
    if min_ == max_ :
        print('all losses are the same')
        return [1/len(losses) for i in losses]
    losses /= (max_ - min_)
    losses *= -1 #higher probability to lower loss
    #
    probs = softmax(losses)
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

def scheduleCompressionCostIncrease(timeFactor):
    if timeFactor >=1:
        costFactor =  1
    else:
        costFactor = 560.7 * np.exp(-9.43*timeFactor)+0.955
    return costFactor

