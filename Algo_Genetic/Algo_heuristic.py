from Interface.generateModel import generateModel
from Interface.Model_options import Model_options




'''generate Model with its options'''
modelOptions = Model_options('RollerCoaster') #type: 'Refinery' , 'MIS' or 'RollerCoaster'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)

heuristicAction = model.projectModel.getHeuristicBestAction()
zeroAction = model.projectModel.getZeroAction()
action = zeroAction
#action = heuristicAction
print("action: " + str(action))
performance = model.projectModel.simulateMean(action)
print('heuristic performance: ' + str(performance))