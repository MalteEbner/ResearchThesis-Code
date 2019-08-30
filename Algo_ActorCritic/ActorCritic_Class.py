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
        outputs,losses = ActorCritic_general.generateActionOutputLayer(self.actionSpace,intLayer)

        #define model
        model = Model(inputs=inputs,outputs=outputs)
        sgd = optimizers.SGD(lr=1)
        model.compile(optimizer=sgd,
                      loss=losses,
                      metrics=['accuracy'])


        self.model = model

        #plot model with graphviz
        plot_model(model, to_file='model.png',show_shapes=True)


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

        action = ActorCritic_general.predictionToAction(outputPrediction,self.actionSpace,kind)
        return action

    def sampleAction(self,input=0):
        nextAction = self.getAction('random',input)
        return nextAction

    def getBestAction(self,input=0):
        bestAction = self.getAction('best', input)
        return bestAction











