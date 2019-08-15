from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from keras.backend import constant
from Meta_Model import ActionSpace
import numpy as np
from tensorflow.keras import optimizers
from keras.utils import to_categorical
from keras.backend import expand_dims




'''provides a policy as neural network with two functions:
i) the policy can be updated given a (state,action)-tuple and a gradient weight (i.e. the advantage)
ii) the policy can be applied on a state to sample a random action given that state
'''


class Policy:
    def __init__(self,actionSpace,stateSpace=0):
        if stateSpace==0:
            self.inputShape = (1,1)
        else:
            pass ##Much more complex input shape
        self.actionSpace = actionSpace
        self.defineModel()



    def defineModel(self):
        categoricalOutputs = self.actionSpace.VariantNumbers()
        if self.actionSpace.withScheduleCompression:
            noRealOutputs = self.actionSpace.noActivities
        else:
            noRealOutputs = 0


        '''Model definition of neural network'''
        #input layer
        inputs = Input(shape=self.inputShape)
        #intermediate layers
        x = Dense(1, activation='relu')(inputs)
        #x = Dense(64, activation='relu')(x)

        #output layers with losses
        outputs = []
        losses = []
        for noVariants in categoricalOutputs:
            variantLayer = Dense(noVariants,activation='softmax')(x)
            outputs.append(variantLayer)
            losses.append('categorical_crossentropy')
        for i in range(noRealOutputs):
            scheduleCompressionLayer = Dense(1)(x)
            outputs.append(scheduleCompressionLayer)
            losses.append('mean_squared_error')

        #define model
        model = Model(inputs=inputs,outputs=outputs)
        sgd = optimizers.SGD(lr=0.1)
        model.compile(optimizer=sgd,
                      loss=losses,
                      metrics=['accuracy'])

        self.model = model


    def updateModel(self,outputActions,updateWeights,input=0):
        noSamples = len(outputActions)
        noOutputs = int(outputActions[0].actionSpace.NoAllVariables())
        if input == 0:
            inputs = np.ones((noSamples,1,1))
        outputs = self.oneHotEncode(outputActions)
        sampleWeights =[updateWeights]*noOutputs
        self.model.fit([inputs],outputs,sample_weight=sampleWeights,verbose=False)


    def oneHotEncode(self,actionList):
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



    def getAction(self,kind,input):
        if input == 0:
            input = np.ones((1,1,1))
        #output is a probability distribution for all categorical outputs
        outputPrediction = self.model.predict(input)
        outputList = [i[0][0] for i in outputPrediction]
        output = outputList

        activityVariantIndizes = []
        for i in range(self.actionSpace.noActivities):
            variantProbs = output[i]
            if kind == 'random':
                chosenVariant = np.random.choice(range(len(variantProbs)),1,p=variantProbs)[0]
            elif kind == 'best':
                chosenVariant = np.argmax(variantProbs)
            activityVariantIndizes.append(chosenVariant)
        eventVariantIndizes = []
        for i in range(self.actionSpace.noEvents):
            variantProbs = output[i+self.actionSpace.noActivities]
            if kind == 'random':
                chosenVariant = np.random.choice(range(len(variantProbs)),1,p=variantProbs)[0]
            elif kind == 'best':
                chosenVariant = np.argmax(variantProbs)
            eventVariantIndizes.append(chosenVariant)

        scheduleCompressionFactors = output[self.actionSpace.noActivities+self.actionSpace.noEvents:]
        scheduleCompressionFactors = [i[0] for i in scheduleCompressionFactors]
        if kind == 'random':
            scheduleCompressionFactors += np.random.normal(0,0.5,len(scheduleCompressionFactors))
        scheduleCompressionFactors = np.maximum(scheduleCompressionFactors,0.5)
        scheduleCompressionFactors = np.minimum(scheduleCompressionFactors,1)


        newAction = ActionSpace.Action(self.actionSpace)
        newAction.saveDirectly(activityVariantIndizes,eventVariantIndizes,scheduleCompressionFactors)
        return newAction

    def getNextAction(self,input=0):
        nextAction = self.getAction('random',input)
        return nextAction

    def getBestAction(self,input=0):
        bestAction = self.getAction('best', input)
        return bestAction











