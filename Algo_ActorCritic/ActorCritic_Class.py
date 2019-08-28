from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.math import maximum as tf_maximum
from tensorflow.math import minimum as tf_minimum
from Interface import ActionSpace
import numpy as np
from tensorflow.keras import optimizers
from keras.initializers import Constant
from tensorflow.keras.utils import plot_model
from Algo_ActorCritic import ActorCritic_general





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
        '''Model definition of neural network'''
        #input layer
        inputs = Input(shape=self.inputShape)
        #intermediate layers
        intLayer = Dense(1, activation='relu')(inputs)
        #intLayer = Dense(64, activation='relu')(intLayer)

        #define output layer
        outputs,losses = ActorCritic_general.generateOutputLayer(self.actionSpace,intLayer)

        #define model
        model = Model(inputs=inputs,outputs=outputs)
        sgd = optimizers.SGD(lr=1)
        model.compile(optimizer=sgd,
                      loss=losses,
                      metrics=['accuracy'])


        self.model = model

        #plot model with graphviz
        plot_model(model, to_file='model.png')


    def update(self,outputActions,updateWeights,input=0):
        noSamples = len(outputActions)
        noOutputs = int(outputActions[0].actionSpace.NoAllVariables())
        noScheduleCompressionFactors = len(outputActions[0].scheduleCompressionFactors)
        if input == 0:
            inputs = np.ones((noSamples,1,1))
        outputs = ActorCritic_general.oneHotEncode(outputActions)
        sampleWeights =[updateWeights]*(noOutputs-noScheduleCompressionFactors) + [np.maximum(updateWeights,0)]*noScheduleCompressionFactors
        self.model.train_on_batch([inputs],outputs,sample_weight=sampleWeights)


    def getAction(self,kind,input):
        if input == 0:
            input = np.ones((1,1,1))
        #output is a probability distribution for all categorical outputs
        outputPrediction = self.model.predict(input)
        outputList = [np.squeeze(i) for i in outputPrediction]
        output = outputList

        activityVariantIndizes = []
        for i ,varNum in enumerate(self.actionSpace.activityVariantNumbers):
            variantProbs = output[i]
            if varNum>1:
                if kind == 'random':
                    variantProbs[0] = 1 - sum(variantProbs[1:])  # ensure variantProbs sums up to 1
                    chosenVariant = np.random.choice(range(len(variantProbs)),1,p=variantProbs)[0]
                elif kind == 'best':
                    chosenVariant = np.argmax(variantProbs)
            else:
                chosenVariant = 0
            activityVariantIndizes.append(chosenVariant)

        eventVariantIndizes = []
        for i ,varNum in enumerate(self.actionSpace.eventVariantNumbers):
            variantProbs = output[i+self.actionSpace.noActivities]
            if varNum>1:
                if kind == 'random':
                    variantProbs[0] = 1 - sum(variantProbs[1:])  # ensure variantProbs sum up to 1
                    chosenVariant = np.random.choice(range(len(variantProbs)),1,p=variantProbs)[0]
                elif kind == 'best':
                    chosenVariant = np.argmax(variantProbs)
            else:
                chosenVariant = 0
            eventVariantIndizes.append(chosenVariant)

        scheduleCompressionFactors = output[self.actionSpace.noActivities+self.actionSpace.noEvents:]
        if kind == 'random':
            scheduleCompressionFactors += np.random.uniform(-0.1,0.1,len(scheduleCompressionFactors))
        scheduleCompressionFactors = [i.item(0) for i in scheduleCompressionFactors]
        scheduleCompressionFactors = np.maximum(scheduleCompressionFactors,0.5)
        scheduleCompressionFactors = np.minimum(scheduleCompressionFactors,1)


        newAction = ActionSpace.Action(self.actionSpace)
        newAction.saveDirectly(activityVariantIndizes,eventVariantIndizes,scheduleCompressionFactors)
        return newAction

    def sampleAction(self,input=0):
        nextAction = self.getAction('random',input)
        return nextAction

    def getBestAction(self,input=0):
        bestAction = self.getAction('best', input)
        return bestAction











