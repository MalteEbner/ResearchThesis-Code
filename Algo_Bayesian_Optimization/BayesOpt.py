from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model
from Meta_Model import commonFunctions

#model = RollerCoaster_Model.Model_RollerCoaster()
#model = Refinery_Model.Model_Refinery()
model = MIS_Model.Model_MIS()#doesn't work because of too many variables


'''
define search spaces
'''

noActivityVariants = model.getVariantNumbers()
space_equalChoice = {}
for num in range(len(noActivityVariants)):
    varName = "act_ID_"+str(num)
    noVariants = noActivityVariants[num]
    space_equalChoice[varName] = hp.choice(varName,range(0,noVariants))

space_softmaxChoice = {}
activityLosses = model.getActivityLosses()
for num in range(len(noActivityVariants)):
    varName = "act_ID_" + str(num)
    noVariants = noActivityVariants[num]
    variantLosses = activityLosses[num]
    variantProbabilities = commonFunctions.probsFromLosses(variantLosses)
    choices = zip(variantProbabilities,range(0,noVariants))
    space_softmaxChoice[varName] = hp.pchoice(varName,choices)

'''
translate chosenVariantIndizes as dictionary into chosenVariantIndizes as list of integers
'''
def get_chosenVariantIndizes(chosenVariantIndizes_dict):
    chosenVariantIndizes = []
    for num in range(len(noActivityVariants)):
        varName = "act_ID_" + str(num)
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