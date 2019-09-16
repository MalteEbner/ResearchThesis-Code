'''Model Imports'''
from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
from Interface import ActionSpace

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model
from Model_general import commonFunctions
from gym import spaces


'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
#modelOptions.interface = "VAE"
model = generateModel(modelOptions)


'''
define search spaces
'''
actionSpace = model.getActionSpace()

searchSpace = {}
varNumber = 0
for space in actionSpace.spaces:
    if isinstance(space, spaces.MultiDiscrete):
        for index, noVariants in enumerate(space.nvec):
            varName = str(varNumber)
            param = hp.choice(varName,range(0,noVariants))
            searchSpace[varName] = param
            varNumber+=1
    elif isinstance(space, spaces.Box):
        for index in range(space.shape[0]):
            varName = varName = str(varNumber)
            param = hp.uniform(varName, 0.5, 1)
            searchSpace[varName] = param
            varNumber += 1
    else:
        raise NotImplementedError

'''
objective Function with variable translation
'''
def varDictToAction(varDict):
    vars = [varDict[str(key)] for key in sorted(varDict.keys(),key = int)]
    action = ActionSpace.Action(actionSpace)
    action.saveEverythingCombined(vars)
    return action

def objectiveFunction(varDict):
    action = varDictToAction(varDict)
    loss = model.simulate_returnLoss(action)
    return loss

'''
do the optimization
'''
best = fmin(objectiveFunction,searchSpace,algo=tpe.suggest,max_evals=40)


bestAction = varDictToAction(best)
model.printAllAboutAction(bestAction)
