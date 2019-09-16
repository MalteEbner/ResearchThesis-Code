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
from gym import spaces





'''provides a policy as neural network with two functions:
i) the policy can be updated given a (state,action)-tuple and a gradient weight (i.e. the advantage)
ii) the policy can be applied on a state to sample a random action given that state
'''


class Policy:
    def __init__(self,actionSpace,stateSpace=0):
        if stateSpace==0:
            self.inputShapeState = (1,)
        else:
            pass ##Much more complex input shape
        self.actionSpace = actionSpace
        self.defineModel()



    def defineModel(self):
        verbose = False
        '''Model definition of neural network'''
        #input layer
        inputs = Input(shape=self.inputShapeState)
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

        if verbose:
            #plot model with graphviz
            plot_model(model, to_file='model.png',show_shapes=True)


    def update(self,outputActions,updateWeights,input=0):
        noSamples = len(outputActions)
        actionSpace = outputActions[0].actionSpace
        sampleWeights = []
        for space in actionSpace.spaces:
            if isinstance(space, spaces.MultiDiscrete):
                sampleWeights += [updateWeights]*len(space.nvec)
            elif isinstance(space, spaces.Box):
                sampleWeights += [np.maximum(updateWeights,0)]
            else:
                raise NotImplementedError
        if input == 0:
            inputs = np.ones((noSamples,1))
        outputs = ActorCritic_general.oneHotEncode(outputActions)
        self.model.train_on_batch([inputs],outputs,sample_weight=sampleWeights)

    def update_Binary(self,outputActions,updateWeights):
        noSamples = len(outputActions)
        actionSpace = outputActions[0].actionSpace

        sampleWeights=updateWeights

        outputActions = [action for action,weight in zip(outputActions,updateWeights) if weight > 0]
        outputs = ActorCritic_general.oneHotEncode(outputActions)

        if input == 0:
            inputs = np.ones((noSamples,1))
        self.model.train_on_batch([inputs],outputs,sample_weight=sampleWeights)


    def getAction(self,kind,inputState):
        if inputState == 0:
            inputState = np.ones((1,1))
        #output is a probability distribution for all categorical outputs
        outputPrediction = self.model.predict(inputState)
        if len(outputPrediction[0].shape)==1:
            outputPrediction = [outputPrediction]
        action = ActorCritic_general.predictionsToActions(outputPrediction,self.actionSpace,kind)
        return action[0]

    def sampleAction(self,inputState=0):
        nextAction = self.getAction('random',inputState=inputState)
        return nextAction

    def getBestAction(self,inputState=0):
        bestAction = self.getAction('best', inputState)
        return bestAction











