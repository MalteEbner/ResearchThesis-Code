'''Model Imports'''
from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
from Interface import ActionSpace
from gym import spaces

'''Optimization imports'''
### Necessary imports
import time
import numpy as np
from emukit.core import  ParameterSpace, CategoricalParameter, OneHotEncoding, ContinuousParameter
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop


'''generate Model with its options'''
modelOptions = Model_options('MIS') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = True
modelOptions.withScheduleCompression=True
model = generateModel(modelOptions)


'''define parameter space'''
actionSpace = model.getActionSpace()
action = ActionSpace.Action(actionSpace)
parameterList = []
encodingList = []
groupNumber = 0
for space in actionSpace.spaces:
    if isinstance(space, spaces.MultiDiscrete):
        for index, noVariants in enumerate(space.nvec):
            options = range(noVariants)
            encoding = OneHotEncoding(options)
            encodingList.append(encoding)
            categoricalParam = CategoricalParameter('Param_%d_%d' % (groupNumber,index), encoding)
            parameterList.append(categoricalParam)
    elif isinstance(space, spaces.Box):
        for index in range(space.shape[0]):
            realParam = ContinuousParameter('Param_%d_%d' % (groupNumber,index),0.5,1)
            parameterList.append(realParam)
    else:
        raise NotImplementedError
    groupNumber +=1
space = ParameterSpace(parameterList)



def encodingToInteger(row, encodingList):
    index = 0
    integers = []
    for encoding in encodingList:
        size = encoding.dimension
        oneHotVector = row[index:index+size]
        index += size
        integ = encoding.get_category(oneHotVector)
        integers.append(integ)
    values = list(integers) + list(row[index:])
    return values

'''define objective function for emukit'''
def emukit_friendly_objective_function(input_rows):
    losses = []
    for row in input_rows:
        values = encodingToInteger(row, encodingList)
        action.saveEverythingCombined(values)
        loss = model.simulate_returnLoss(action)
        print('loss: ' + str(loss))
        losses.append([loss])
    return np.array(losses)

'''use random forests as model'''
from emukit.examples.models.random_forest import RandomForest
from emukit.experimental_design import RandomDesign

random_design = RandomDesign(space)
initial_points_count = 3
X_init = random_design.get_samples(initial_points_count)
Y_init = emukit_friendly_objective_function(X_init)

rf_model = RandomForest(X_init, Y_init)

'''start finding optimimum'''
loop = BayesianOptimizationLoop(space,rf_model)

start = time.time()
loop.run_loop(emukit_friendly_objective_function, 10)
end = time.time()
print('time needed: ' + str(end-start))

'''get results'''
bestIteration = np.argmin(loop.loop_state.Y)
bestPointEncoded = loop.loop_state.X[bestIteration]
bestPoint = encodingToInteger(bestPointEncoded,encodingList)
bestLoss = loop.loop_state.Y[bestIteration]
print('best: loss: ' + str(bestLoss) + ' point: ' + str(bestPoint))

print("end of optimization")
