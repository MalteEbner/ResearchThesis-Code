from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Add
from tensorflow.math import maximum as tf_maximum
from tensorflow.math import minimum as tf_minimum
from Interface import ActionSpace
import numpy as np
from tensorflow.keras import optimizers
from keras.initializers import Constant
from tensorflow.keras.utils import plot_model
from Algo_ActorCritic import ActorCritic_general
from tensorflow.keras import backend as K
from Algo_ActorCritic import ActorCritic_general
from Algo_ActorCritic.ActorCritic_Class import Policy
from tensorflow.keras.losses import mse, categorical_crossentropy
from Interface_VAE import VAE







'''provides a policy as neural network with two functions:
i) the policy can be updated given a (state,action)-tuple and a gradient weight (i.e. the advantage)
ii) the policy can be applied on a state to sample a random action given that state
'''


class VAE_Policy(Policy):
    def __init__(self,actionSpace,stateSpace=0):
        self.latentDim = 4
        self.VAE = VAE.VAE_Model(actionSpace,self.latentDim)

        super().__init__(actionSpace, stateSpace)

    def defineModel(self):
        '''Model definition of neural network mapping from state to action'''
        # input layer
        inputState = Input(shape=self.inputShapeState)
        # intermediate layers
        intLayer = Dense(1, activation='relu')(inputState)
        # intLayer = Dense(64, activation='relu')(intLayer)

        # define output layer to latent action
        latentAction = Dense(self.latentDim)(intLayer)
        noiseLayer = Input(self.latentDim)
        noisyLatentAction = Add()([latentAction,noiseLayer])

        # from latent action to action
        action = self.VAE.decoder(noisyLatentAction)

        # define model
        inputs = [inputState,noiseLayer]
        model = Model(inputs=inputs, outputs=action)
        #model.layers[-1].trainable=False #don't train decoder

        sgd = optimizers.SGD(lr=1)
        model.compile(optimizer=sgd,
                      loss='mse',
                      metrics=['accuracy'])

        self.model = model

        # plot model with graphviz
        plot_model(model, to_file='model_ActorCritic_VAE.png', show_shapes=True)

        '''model definition of completete policy mapping from state to action'''

    def getAction(self, kind, inputState):
        predictedAction = self.getPrediction(kind,inputState)
        action = ActorCritic_general.predictionToAction(predictedAction,self.actionSpace,kind)
        return action

    def getPrediction(self,kind='best',inputState=0):
        inputState = self.getStateOrDefault(inputState)
        if kind == 'random':
            input = [inputState, self.getGaussianNoise()]
        else:
            input = [inputState, self.getZeroNoise()]
        predictedAction = self.model.predict(input)
        return predictedAction

    def getStateOrDefault(self,inputState,batchSize=1):
        if inputState == 0:
            if batchSize == 1:
                inputState = np.ones(self.inputShapeState)
            else:
                inputState = np.ones((batchSize,)+self.inputShapeState)
        return inputState

    def getZeroNoise(self,batchSize=1):
        zero = np.zeros((batchSize,self.latentDim))
        return zero

    def getGaussianNoise(self,batchSize=1):
        noise = K.random_normal(shape=(batchSize, self.latentDim))
        return noise

    #update both the policy (state -> latentAction -> action) and VAE(action -> latentAction -> action)
    def update(self,outputActions,updateWeights,learningRate=1,inputState=0):
        batchSize = len(outputActions)
        inputState = self.getStateOrDefault(inputState,batchSize)
        #update policy
        noOutputs = int(outputActions[0].actionSpace.NoAllVariables())
        noScheduleCompressionFactors = len(outputActions[0].scheduleCompressionFactors)
        encodedActions = ActorCritic_general.oneHotEncode(outputActions)
        sampleWeights =[updateWeights]*(noOutputs-noScheduleCompressionFactors) + [np.maximum(updateWeights,0)]*noScheduleCompressionFactors
        input = [inputState,self.getZeroNoise(batchSize)]
        self.model.train_on_batch(input,encodedActions,sample_weight=sampleWeights)
        #update VAE
        self.VAE.update(encodedActions,learningRate)















