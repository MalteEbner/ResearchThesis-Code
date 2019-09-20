import Interface_VAE
from Interface.Model_options import Model_options
from Interface.generateModel import generateModel
from Interface_VAE.VAE import VAE_sampling
import numpy as np
from tensorflow.keras.models import load_model


'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'Refinery' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)
projectModel = model.projectModel




def generatePretrainedVAEModel(projectModel,latentDim):
    actionSpace = projectModel.getActionSpace()
    vae = Interface_VAE.VAE.VAE_Model(projectModel,latentDim,False)


    print("starting pre-training of VAE:")


    # get threshhold for good actions as mean minus standard deviation averaged with best of many random actions
    losses = [projectModel.simulate_returnLoss(actionSpace.sample()) for i in range(1000)]
    baseline = np.mean(losses)
    baseStd = np.std(losses)
    bestLoss = np.min(losses)
    threshhold = np.mean([baseline - baseStd, bestLoss])
    print('threshhold: ' + str(threshhold))

    if True:
        # pre-train with best of following actions
        noIters = 100
        noSamplesPerIter = 16*1024
        noSamplesTrainedOnTotal = 0
        for i in range(noIters):
            actions = [actionSpace.sample() for i in range(noSamplesPerIter)]
            goodActions = [action for action in actions if projectModel.simulate_returnLoss(action) < threshhold]
            noSamplesTrainedOn = len(goodActions)
            noSamplesTrainedOnTotal += noSamplesTrainedOn
            if noSamplesTrainedOn>0:
                vae.update(goodActions, 0.01)
            if i % 1 == 0:
                print("iteration %d: trained on %d of %d samples (%f%%)" % (
                i, noSamplesTrainedOn, noSamplesPerIter, noSamplesTrainedOn * 100. / noSamplesPerIter))

        print("end pre-training of VAE")
        print("trained in total on %d of %d samples (%f%%)" % (noSamplesTrainedOnTotal, noSamplesPerIter * noIters,
                                                               noSamplesTrainedOnTotal * 100. / (noSamplesPerIter * noIters)))


    # save trained model
    name = projectModel.modelOptions.asPretrainedVAE_Filename(projectModel)
    vae.model.save(name)

    #test if model can be loaded
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform

    model = load_model(name)
    print('loaded successfully')

'''generate and pre-train VAE'''
latentDim = 16
generatePretrainedVAEModel(projectModel,latentDim)