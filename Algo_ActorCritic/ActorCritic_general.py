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
            for noVariants in space.nvec:
                variantLayer = Input(shape=(noVariants,))
                inputs.append(variantLayer)
        elif isinstance(space,spaces.Box):
                scheduleLayer = Input(shape=space.shape)
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
            for noVariants in space.nvec:
                variantLayer = Dense(noVariants, activation='softmax')(previousLayer)
                outputs.append(variantLayer)
                losses.append('categorical_crossentropy')
        elif isinstance(space,spaces.Box):
            # output should range in box constraints
            if space.is_bounded():
                boxMean = np.mean([space.low,space.high],axis=0)
                scheduleCompressionLayer_unconstrained = Dense(space.shape[0], bias_initializer=Constant(value=boxMean))(
                    previousLayer)
                scheduleCompressionLayer = Lambda(lambda x: tf_minimum(tf_maximum(x, space.low), space.high))(
                    scheduleCompressionLayer_unconstrained)
            else:#assumes space is unbounded on both sides
                scheduleCompressionLayer = Dense(space.shape[0])(previousLayer)
            outputs.append(scheduleCompressionLayer)
            losses.append('mean_squared_error')
        else:
            raise NotImplementedError
    return outputs,losses


def softmaxPredictionListToChoices(predictionList,kind='best'):
    chosenVariants =[]

    try:
        for variantProbs in predictionList:
            if isinstance(variantProbs,list) and len(variantProbs>1):
                varNum = len(variantProbs)
                if kind == 'random':
                    variantProbs[0] = 1 - sum(variantProbs[1:])  # ensure variantProbs sums up to 1
                    chosenVariant = np.random.choice(range(varNum), 1, p=variantProbs)[0]
                elif kind == 'best':
                    chosenVariant = np.argmax(variantProbs)
            else:
                chosenVariant = 0
            chosenVariants.append(chosenVariant)
    except ValueError:
        print('ERROR: variantProbs either NaN or non-negative:')
        print(str(variantProbs))
        raise ValueError

    return chosenVariants

def oneHotEncode(actionList):
    noActivities = actionList[0].actionSpace.noActivities
    VariantNumbers = actionList[0].actionSpace.VariantNumbers()
    variableListList = [
    np.concatenate((action.activityIndizes, action.eventIndizes, action.scheduleCompressionFactors)) for action in actionList]
    variables = np.array(variableListList)
    outputs = []
    for i in range(len(VariantNumbers)):
        noVariants = VariantNumbers[i]
        encoding = to_categorical(variables[:,i],num_classes=noVariants)
        #encoding = expand_dims(encoding,axis=1)
        outputs.append(encoding)
    if actionList[0].actionSpace.withScheduleCompression:
        for i in range(noActivities):
            encoding = variables[:,len(VariantNumbers)+i]
            outputs.append(encoding)
    return outputs


def predictionToAction(prediction,actionSpace,kind):
    raise NotImplementedError
    '''
    outputList = [np.squeeze(i) for i in prediction]
    output = outputList
    action = ActionSpace.Action(actionSpace)

    activityVariantProbs = output[:actionSpace.noActivities]
    eventVariantProbs = output[actionSpace.noActivities:actionSpace.noActivities+actionSpace.noEvents]

    activityVariantIndizes = softmaxPredictionListToChoices(activityVariantProbs)
    eventVariantIndizes = softmaxPredictionListToChoices(eventVariantProbs)

    scheduleCompressionFactors = output[actionSpace.noActivities + actionSpace.noEvents:]
    if kind == 'random':
        scheduleCompressionFactors += np.random.uniform(-0.1, 0.1, len(scheduleCompressionFactors))
    scheduleCompressionFactors = [i.item(0) for i in scheduleCompressionFactors]
    scheduleCompressionFactors = np.maximum(scheduleCompressionFactors, 0.5)
    scheduleCompressionFactors = np.minimum(scheduleCompressionFactors, 1)

    newAction = ActionSpace.Action(actionSpace)
    newAction.saveDirectly(activityVariantIndizes, eventVariantIndizes, scheduleCompressionFactors)
    return newAction
    '''
