from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Reshape, Flatten
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


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
# taken from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def VAE_sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



'''provides a policy as neural network with two functions:
i) the policy can be updated given a (state,action)-tuple and a gradient weight (i.e. the advantage)
ii) the policy can be applied on a state to sample a random action given that state
'''


class VAE_Policy(Policy):
    def __init__(self,actionSpace,stateSpace=0):
        super().__init__(actionSpace,stateSpace)
        #self.inputShape = (1)

    def defineModel(self):
        verbose = False

        self.latentDim = 4
        latentDim = self.latentDim

        #define encoder model
        encoder_inputLayers = ActorCritic_general.generatActionInputLayer(self.actionSpace)
        encoder_inputConcatLayer = concatenate(encoder_inputLayers)
        encoder_intLayer = Dense(latentDim*4,activation='relu')(encoder_inputConcatLayer)
        encoder_intLayer_last = Dense(latentDim*2,activation='relu')(encoder_intLayer)
        encoder_meanLayer = Dense(latentDim,activation='relu')(encoder_intLayer_last)
        encoder_logVarianceLayer = Dense(latentDim, activation='relu')(encoder_intLayer_last)
        encoder_outputLayer = Lambda(VAE_sampling,output_shape=(latentDim,))([encoder_meanLayer,encoder_logVarianceLayer])
        encoder_outputLayers = [encoder_meanLayer,encoder_logVarianceLayer,encoder_outputLayer]
        encoder_Model = Model(encoder_inputLayers,encoder_outputLayers,name='encoder')


        if verbose:
            encoder_Model.summary()
            plot_model(encoder_Model, to_file='vae_encoder.png', show_shapes=True)

        #define decoder model
        decoder_inputLayerLatentAction = Input(shape=(latentDim,))
        decoder_inputLayerState = Input(shape=self.inputShape)
        decoder_inputLayers = [decoder_inputLayerLatentAction,decoder_inputLayerState]
        decoder_inputConcatLayer = concatenate(decoder_inputLayers,axis = 1)
        decoder_intLayer = Dense(latentDim*2,activation='relu')(decoder_inputConcatLayer)
        decoder_intLayer_last = Dense(latentDim*4, activation='relu')(decoder_intLayer)
        decoder_outputLayer, losses_reconstruction = ActorCritic_general.generateActionOutputLayer(self.actionSpace,decoder_intLayer_last)
        decoder_Model = Model(decoder_inputLayers,decoder_outputLayer,name='decoder')
        if verbose:
            decoder_Model.summary()
            plot_model(decoder_Model, to_file='vae_decoder.png', show_shapes=True)

        #define VAE model

        latentActionLayer = encoder_outputLayer
        outputs = decoder_Model([latentActionLayer,decoder_inputLayerState])
        vae_model = Model([encoder_inputLayers,decoder_inputLayerState],outputs,name='vae')
        if verbose:
            vae_model.summary()
            plot_model(vae_model, to_file='vae_model.png', show_shapes=True)

        #add KL-divergence to losses
        pass

        #define model
        sgd = optimizers.SGD(lr=1)
        vae_model.compile(optimizer=sgd,
                      loss=losses_reconstruction,
                      metrics=['accuracy'])

        #save models
        self.model = vae_model
        self.encoder = encoder_Model
        self.decoder = decoder_Model

    def getAction(self,kind,inputState=0,explorationFactor=0):
        if inputState == 0:
            inputState = np.ones((1,1))

        #introduce randomness through latent space
        inputLatentAction = np.random.rand(1,self.latentDim)
        inputLatentAction *= explorationFactor
        if explorationFactor<0:
            kind = 'random'
        else:
            kind = 'best'

        totalInput = [inputLatentAction,inputState]
        outputPrediction = self.decoder.predict(totalInput)
        action = ActorCritic_general.predictionToAction(outputPrediction,self.actionSpace,kind)
        return action

    def update(self,outputActions,updateWeights,inputStates=0):
        noSamples = len(outputActions)
        noOutputs = int(outputActions[0].actionSpace.NoAllVariables())
        noScheduleCompressionFactors = len(outputActions[0].scheduleCompressionFactors)
        if inputStates == 0:
            inputsStates = np.ones((noSamples,1))
        encodedActions = ActorCritic_general.oneHotEncode(outputActions)
        outputs = encodedActions
        inputs = encodedActions + [inputsStates]
        sampleWeights =[updateWeights]*(noOutputs-noScheduleCompressionFactors) + [np.maximum(updateWeights,0)]*noScheduleCompressionFactors
        self.model.train_on_batch(inputs,outputs,sample_weight=sampleWeights)






