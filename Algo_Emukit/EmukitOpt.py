'''Model Imports'''
from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model
from Meta_Model import commonFunctions

model = RollerCoaster_Model.Model_RollerCoaster()
#model = Refinery_Model.Model_Refinery()
#model = MIS_Model.Model_MIS()


'''Optimization imports'''
### Necessary imports
import tensorflow as tf
import GPy
import numpy as np

from emukit.core import  ParameterSpace, CategoricalParameter, OneHotEncoding
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop


'''define parameter space'''
noActivityVariants = model.getVariantNumbers()
parameterList = []
encodingList = []
for index,varNumber in enumerate(noActivityVariants):
    options = range(varNumber)
    encoding = OneHotEncoding(options)
    encodingList.append(encoding)
    categoricalParam = CategoricalParameter('paramName_'+str(index),encoding)
    parameterList.append(categoricalParam)
space = ParameterSpace(parameterList)



def encodingToInteger(row, encodingList):
    index = 0
    integers = []
    for encoding in encodingList:
        size = encoding.dimension
        oneHotVector = row[index:index+size]
        index+=size
        integ = encoding.get_category(oneHotVector)
        integers.append(integ)
    return integers

'''define objective function for emukit'''
def emukit_friendly_objective_function(input_rows):
    losses = []
    for row in input_rows:
        chosenOptionIndizes = encodingToInteger(row, encodingList)
        loss = model.simulate_returnLoss(chosenOptionIndizes)
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
loop.run_loop(emukit_friendly_objective_function, 10)

'''get results'''
bestIteration = np.argmin(loop.loop_state.Y)
bestPointEncoded = loop.loop_state.X[bestIteration]
bestPoint = encodingToInteger(bestPointEncoded,encodingList)
bestLoss = loop.loop_state.Y[bestIteration]
print('best: loss: ' + str(bestLoss) + ' point: ' + str(bestPoint))

print("end of optimization")
