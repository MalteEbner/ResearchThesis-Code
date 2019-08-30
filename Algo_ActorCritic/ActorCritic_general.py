from Interface import ActionSpace
from tensorflow.keras.layers import Input, Dense, Lambda
from keras.initializers import Constant
from tensorflow.math import maximum as tf_maximum
from tensorflow.math import minimum as tf_minimum
from keras.utils import to_categorical
from keras.backend import expand_dims
import numpy as np

def generatActionInputLayer(actionSpace):
    categoricalOutputs = actionSpace.VariantNumbers()
    if actionSpace.withScheduleCompression:
        noRealOutputs = actionSpace.noActivities
    else:
        noRealOutputs = 0
        
    # input Layers
    inputs = []
    for noVariants in categoricalOutputs:
        variantLayer = Input(shape=(noVariants,))
        inputs.append(variantLayer)
    for i in range(noRealOutputs):
        # output should range in 0.5 and 1.0
        scheduleCompressionLayer = Input(shape=(1,))
        inputs.append(scheduleCompressionLayer)
    return inputs

def generateActionOutputLayer(actionSpace,previousLayer):
    categoricalOutputs = actionSpace.VariantNumbers()
    if actionSpace.withScheduleCompression:
        noRealOutputs = actionSpace.noActivities
    else:
        noRealOutputs = 0


    # output layers with losses
    outputs = []
    losses = []
    for noVariants in categoricalOutputs:
        variantLayer = Dense(noVariants, activation='softmax')(previousLayer)
        outputs.append(variantLayer)
        losses.append('categorical_crossentropy')
    for i in range(noRealOutputs):
        # output should range in 0.5 and 1.0
        scheduleCompressionLayer = Dense(1, bias_initializer=Constant(value=0.75))(previousLayer)
        scheduleCompressionLayer2 = Lambda(lambda x: tf_minimum(tf_maximum(x, 0.5), 1))(scheduleCompressionLayer)
        # scheduleCompressionLayer3 = Lambda(lambda x: x,1))(scheduleCompressionLayer2)
        outputs.append(scheduleCompressionLayer2)
        losses.append('mean_squared_error')

    return outputs, losses

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
        encoding = expand_dims(encoding,axis=1)
        outputs.append(encoding)
    if actionList[0].actionSpace.withScheduleCompression:
        for i in range(noActivities):
            encoding = variables[:,len(VariantNumbers)+i]
            outputs.append(encoding)
    return outputs

def predictionToAction(prediction,actionSpace,kind):
    outputList = [np.squeeze(i) for i in prediction]
    output = outputList

    activityVariantIndizes = []
    for i, varNum in enumerate(actionSpace.activityVariantNumbers):
        variantProbs = output[i]
        if varNum > 1:
            if kind == 'random':
                variantProbs[0] = 1 - sum(variantProbs[1:])  # ensure variantProbs sums up to 1
                chosenVariant = np.random.choice(range(len(variantProbs)), 1, p=variantProbs)[0]
            elif kind == 'best':
                chosenVariant = np.argmax(variantProbs)
        else:
            chosenVariant = 0
        activityVariantIndizes.append(chosenVariant)

    eventVariantIndizes = []
    for i, varNum in enumerate(actionSpace.eventVariantNumbers):
        variantProbs = output[i + actionSpace.noActivities]
        if varNum > 1:
            if kind == 'random':
                variantProbs[0] = 1 - sum(variantProbs[1:])  # ensure variantProbs sum up to 1
                chosenVariant = np.random.choice(range(len(variantProbs)), 1, p=variantProbs)[0]
            elif kind == 'best':
                chosenVariant = np.argmax(variantProbs)
        else:
            chosenVariant = 0
        eventVariantIndizes.append(chosenVariant)

    scheduleCompressionFactors = output[actionSpace.noActivities + actionSpace.noEvents:]
    if kind == 'random':
        scheduleCompressionFactors += np.random.uniform(-0.1, 0.1, len(scheduleCompressionFactors))
    scheduleCompressionFactors = [i.item(0) for i in scheduleCompressionFactors]
    scheduleCompressionFactors = np.maximum(scheduleCompressionFactors, 0.5)
    scheduleCompressionFactors = np.minimum(scheduleCompressionFactors, 1)

    newAction = ActionSpace.Action(actionSpace)
    newAction.saveDirectly(activityVariantIndizes, eventVariantIndizes, scheduleCompressionFactors)
    return newAction
