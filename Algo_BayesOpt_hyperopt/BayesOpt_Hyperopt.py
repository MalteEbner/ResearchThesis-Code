'''Model Imports'''
from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
from Interface import ActionSpace

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model
from Model_general import commonFunctions



'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = True
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)


'''
define search spaces
'''
actionSpace = model.getActionSpace()
if actionSpace.withScheduleCompression:
    raise NotImplementedError

noActivityVariants = actionSpace.VariantNumbers()
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

def objectiveFunction(chosenVariantIndizes_dict):
    action = ActionSpace.Action(actionSpace)
    action.saveIndizesCombined(get_chosenVariantIndizes(chosenVariantIndizes_dict))
    loss = model.simulate_returnLoss(action)
    return loss

'''
do the optimization
'''
best = fmin(objectiveFunction,space_softmaxChoice,algo=tpe.suggest,max_evals=400)

print("Best: " + str(objectiveFunction(best)))
