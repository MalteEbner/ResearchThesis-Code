from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from Refinery_Model_definition import Model_Refinery
import numpy as np

def softmax(x):
    x = np.array(x)
    y = np.exp(x)/sum(np.exp(x))
    z = np.round(y*(2**10))
    z = z/(2**10)
    z[0] = 1-sum(z[1:])
    return z

model = Model_Refinery()


'''
define search spaces
'''

noActivityVariants = model.getVariantNumbers()
space_equalChoice = {}
for act, noVariants in zip(model.activities,noActivityVariants):
    varName = "act_ID_"+str(act.activity_ID)
    space_equalChoice[varName] = hp.choice(varName,range(0,noVariants))

space_softmaxChoice = {}
activityLosses = model.getActivityLosses(lambda tupl: tupl[0] * 3000 + tupl[1])
for act, noVariants,variantLosses in zip(model.activities,noActivityVariants,activityLosses):
    varName = "act_ID_"+str(act.activity_ID)
    variantProbabilities = softmax(np.array(variantLosses)/float(min(variantLosses))*-19)
    sumProbs = sum(variantProbabilities)
    choices = zip(variantProbabilities,range(0,noVariants))
    space_softmaxChoice[varName] = hp.pchoice(varName,choices)

'''
translate chosenVariantIndizes as dictionary into chosenVariantIndizes as list of integers
'''
def get_chosenVariantIndizes(chosenVariantIndizes_dict):
    chosenVariantIndizes = []
    for act in model.activities:
        varName = "act_ID_" + str(act.activity_ID)
        chosenVariantID = int(chosenVariantIndizes_dict[varName])
        chosenVariantIndizes.append(chosenVariantID)
    return chosenVariantIndizes

'''
do the optimization
'''
optFunction = lambda chosenVariantIndizes_dict: model.simulate_returnLoss(get_chosenVariantIndizes(chosenVariantIndizes_dict))
best = fmin(optFunction,space_softmaxChoice,algo=rand.suggest,max_evals=200)

print(model.simulate(get_chosenVariantIndizes(best)))

best = model.getStartpoint()
print(model.simulate(best))