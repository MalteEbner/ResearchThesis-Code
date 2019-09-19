from Interface import ActionSpace
from tensorflow.keras.layers import Input, Dense, Lambda
from keras.initializers import Constant
from tensorflow.math import maximum as tf_maximum
from tensorflow.math import minimum as tf_minimum
from keras.utils import to_categorical
from keras.backend import expand_dims
import numpy as np
from gym import spaces

def generatActionInputLayer(actionSpace):
    # input Layers
    inputs = []
    for space in actionSpace.spaces:
        if isinstance(space,spaces.MultiDiscrete):
            for ind, noVariants in enumerate(space.nvec):
                variantLayer = Input(shape=(noVariants,))
                inputs.append(variantLayer)
        elif isinstance(space,spaces.Box):
                scheduleLayer = Input(shape=space.shape,name='scheduleCompression')
                inputs.append(scheduleLayer)
        else:
            raise NotImplementedError

    return inputs

def generateActionOutputLayer(actionSpace,previousLayer):

    # output layers with losses
    outputs = []
    losses = []
    for space in actionSpace.spaces:
        if isinstance(space,spaces.MultiDiscrete):
            for ind, noVariants in enumerate(space.nvec):
                variantLayer = Dense(noVariants, activation='softmax')(previousLayer)
                outputs.append(variantLayer)
                losses.append('categorical_crossentropy')
        elif isinstance(space,spaces.Box):
            # output should range in box constraints
            if space.is_bounded():
                boxMean = np.mean([space.low,space.high],axis=0)
                scheduleCompressionLayer_unconstrained = Dense(space.shape[0], bias_initializer=Constant(value=boxMean), name='scheduleCompression_unconstrained')(
                    previousLayer)
                scheduleCompressionLayer = Lambda(lambda x: tf_minimum(tf_maximum(x, space.low), space.high),name='scheduleCompression_in_bounds')(
                    scheduleCompressionLayer_unconstrained)
            else:#assumes space is unbounded on both sides
                scheduleCompressionLayer = Dense(space.shape[0])(previousLayer)
            outputs.append(scheduleCompressionLayer)
            losses.append('mean_squared_error')
        else:
            raise NotImplementedError
    return outputs,losses


def softmaxPredictionListToChoices(predictionNDarray,kind='best'):
    chosenVariants =[]
    if len(predictionNDarray.shape)==1:
        predictionNDarray = np.expand_dims(predictionNDarray,0)
    try:
        for variantProbs in predictionNDarray:
            if hasattr(variantProbs, '__iter__') and len(variantProbs)>1:
                varNum = len(variantProbs)
                if kind == 'random':
                    # ensure variantProbs sums up to 1
                    bestVariant = np.argmax(variantProbs)
                    sumVariantProbs = np.sum(variantProbs)
                    variantProbs[bestVariant] += 1-sumVariantProbs
                    # sample variant
                    chosenVariant = np.random.choice(range(varNum), 1, p=variantProbs)[0]
                elif kind == 'best':
                    chosenVariant = np.argmax(variantProbs)
            else:
                chosenVariant=0
            chosenVariants.append(chosenVariant)
    except ValueError:
        print('ERROR: variantProbs either NaN or non-negative:')
        print(str(variantProbs))
        raise ValueError

    return chosenVariants

def oneHotEncode(actionList):
    actionSpace = actionList[0].actionSpace
    outputs = []
    variableList = [np.concatenate(action.valuesList) for action in actionList]
    variables = np.array(variableList)
    noVarsSoFar = 0
    for space in actionSpace.spaces:
        if isinstance(space,spaces.MultiDiscrete):
            for noVariants in space.nvec:
                encoding = to_categorical(variables[:, noVarsSoFar], num_classes=noVariants)
                # encoding = expand_dims(encoding,axis=1)
                outputs.append(encoding)
                noVarsSoFar +=1
        elif isinstance(space,spaces.Box):
            encoding = variables[:, noVarsSoFar:noVarsSoFar+space.shape[0]]
            # encoding = expand_dims(encoding,axis=1)
            outputs.append(encoding)
            noVarsSoFar += space.shape[0]
        else:
            raise NotImplementedError
    return outputs


def predictionsToActions(predictions,actionSpace,kind):
    noActions = predictions[0].shape[0]
    output = predictions
    actions = [ActionSpace.Action(actionSpace) for i in range(noActions)]
    for action in actions:
        action.valuesList = [[] for i in actionSpace.spaces]


    noVarsSoFar = 0
    for spaceIndex, space in enumerate(actionSpace.spaces):



        if isinstance(space,spaces.MultiDiscrete):
            for variantNum in space.nvec:
                noVars = 1
                values = output[noVarsSoFar:noVarsSoFar + noVars][0]
                indizes = softmaxPredictionListToChoices(values, kind)
                noVarsSoFar += noVars
                for index, action in enumerate(actions):
                    action.valuesList[spaceIndex].append(indizes[index])

        elif isinstance(space,spaces.Box):
            noVars = space.shape[0]
            values = output[noVarsSoFar:noVarsSoFar + noVars]
            values += np.zeros(len(values))
            noVarsSoFar += noVars
            if kind == 'random':
                values += np.random.uniform(-0.1, 0.1, len(values))
                values = np.maximum(space.low,values)
                values = np.minimum(space.high,values)
            values = values[0]
            if len(values.shape)==1:
                values = np.expand_dims(values,0)
            for index, action in enumerate(actions):
                valuesToSave = values[index, :]
                action.valuesList[spaceIndex] += list(valuesToSave)

        else:
            raise NotImplementedError

    return actions
