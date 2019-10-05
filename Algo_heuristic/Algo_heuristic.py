from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
import time



'''generate Model with its options'''
modelOptions = Model_options('RollerCoaster') #type: 'Refinery' , 'MIS' or 'RollerCoaster'
model = generateModel(modelOptions)

heuristicAction = model.projectModel.getHeuristicBestAction()
zeroAction = model.projectModel.getZeroAction()
action = zeroAction
#action = heuristicAction
print("action: " + str(action))#

noIters = 1000
randomTestsToMean = 10
noEvals = noIters*randomTestsToMean
start = time.time()
performance = [model.projectModel.simulateMean(action,randomTestsToMean=randomTestsToMean) for i in range(noIters)][0]
timeDiff = time.time()-start
print("Needed %f seconds for %d evaluations, this equals %fms per evalutation" %(timeDiff,noEvals,timeDiff*1000./noEvals))



print('heuristic performance: ' + str(performance))