from Interface import ActionSpace
from tensorflow.keras.layers import Input, Dense, Lambda
from keras.initializers import Constant
from tensorflow.math import maximum as tf_maximum
from tensorflow.math import minimum as tf_minimum
from keras.utils import to_categorical
from keras.backend import expand_dims
import numpy as np


def generateOutputLayer(actionSpace,previousLayer):
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