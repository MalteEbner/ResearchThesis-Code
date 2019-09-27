from Interface.generateModel import generateModel
from Interface.Model_options import Model_options




'''generate Model with its options'''
modelOptions = Model_options('MIS ') #type: 'Refinery' , 'MIS' or 'RollerCoaster'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)

heuristicAction = model.projectModel.getHeuristicBestAction()
print("action: " + str(heuristicAction))
performance = model.projectModel.simulateMean(heuristicAction)
print('heuristic performance: ' + str(performance))