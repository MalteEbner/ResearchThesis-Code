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



def encodingToInteger(rows, encodingList):
    integerMatrix = []
    for inputPoint in rows:
        index = 0
        integers = []
        for encoding in encodingList:
            size = encoding.dimension
            oneHotVector = inputPoint[index:index+size]
            index+=size
            integ = encoding.get_category(oneHotVector)
            integers.append(integ)
        integerMatrix.append(integers)
    return integerMatrix

'''define objective function for emukit'''
def emukit_friendly_objective_function(input_rows):
    #chosenOptionIndizes = [encodingToInteger(row) for row in input_rows]
    chosenOptionMatrix = encodingToInteger(input_rows,encodingList)
    losses = []
    for chosenOptionIndizes in chosenOptionMatrix:
        loss = model.simulate(chosenOptionIndizes)
        losses.append(loss)
    return np.array(losses)

'''use random forests as model'''
from emukit.examples.models.random_forest import RandomForest
from emukit.experimental_design import RandomDesign

random_design = RandomDesign(space)
initial_points_count = 5
X_init = random_design.get_samples(initial_points_count)
Y_init = emukit_friendly_objective_function(X_init)

rf_model = RandomForest(X_init, Y_init)

'''start finding optimimum'''
loop = BayesianOptimizationLoop(rf_model, space)
loop.run_loop(emukit_friendly_objective_function, 10)

