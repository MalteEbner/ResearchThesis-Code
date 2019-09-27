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

'''Ignore warning'''
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore',category=DataConversionWarning)


'''generate Model with its options'''
modelOptions = Model_options('MIS ') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
#modelOptions.interface = "VAE"
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
        if space.is_bounded():
            for index, low, high in zip(range(space.shape[0]), space.low, space.high):
                realParam = ContinuousParameter('Param_%d_%d' % (groupNumber, index), low, high)
                parameterList.append(realParam)
        else:
            for index in range(space.shape[0]):
                realParam = ContinuousParameter('Param_%d_%d' % (groupNumber, index),-5,5)#no unbounded param possible
                parameterList.append(realParam)

    else:
        raise NotImplementedError
    groupNumber +=1
space = ParameterSpace(parameterList)



def encodingToAction(row, encodingList):
    index = 0
    integers = []
    for encoding in encodingList:
        size = encoding.dimension
        oneHotVector = row[index:index+size]
        index += size
        integ = encoding.get_category(oneHotVector)
        integers.append(integ)
    values = list(integers) + list(row[index:])
    action.saveEverythingCombined(values)
    return action

'''define objective function for emukit'''
noIter = 0
def emukit_friendly_objective_function(input_rows):
    losses = []
    for row in input_rows:
        action = encodingToAction(row, encodingList)
        loss = model.simulate_returnLoss(action)

        global noIter
        noIter +=1
        global start
        timeDiff = time.time()-start
        print('Iter: %d, loss: %f, time: %f' % (noIter,loss,timeDiff))

        losses.append([loss])
    return np.array(losses)

'''use random forests as model'''
from emukit.examples.models.random_forest import RandomForest
from emukit.experimental_design import RandomDesign

random_design = RandomDesign(space)

'''start finding optimimum'''


start = time.time()

initial_points_count = 1500
X_init = random_design.get_samples(initial_points_count)
Y_init = emukit_friendly_objective_function(X_init)
rf_model = RandomForest(X_init, Y_init)
loop = BayesianOptimizationLoop(space,rf_model)

loop.run_loop(emukit_friendly_objective_function, 1500)
end = time.time()
print('time needed: ' + str(end-start))

'''get results'''
bestIteration = np.argmin(loop.loop_state.Y)
bestPointEncoded = loop.loop_state.X[bestIteration]
bestAction = encodingToAction(bestPointEncoded,encodingList)
model.printAllAboutAction(bestAction)

print("end of optimization")
