from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from Refinery_Model import Model_Refinery
import numpy as np
from softmax import probsFromLosses

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
activityLosses = model.getActivityLosses()
for act, noVariants,variantLosses in zip(model.activities,noActivityVariants,activityLosses):
    varName = "act_ID_"+str(act.activity_ID)
    variantProbabilities = probsFromLosses(variantLosses)
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
best = fmin(optFunction,space_softmaxChoice,algo=tpe.suggest,max_evals=400)

print("Best: " + str(model.simulate(get_chosenVariantIndizes(best))))

best = model.getStartpoint()
print("Default: " + str(model.simulate(best)))