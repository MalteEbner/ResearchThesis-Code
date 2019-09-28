import Interface_VAE
from Interface.Model_options import Model_options
from Interface.generateModel import generateModel
from Interface_VAE.VAE import VAE_sampling
import numpy as np
from tensorflow.keras.models import load_model
import time


'''generate Model with its options'''
modelOptions = Model_options('RollerCoaster') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
#modelOptions.probabilistic = True
#modelOptions.withScheduleCompression=True
model = generateModel(modelOptions)
projectModel = model.projectModel
latentDim = 32
if modelOptions.withScheduleCompression:
    latentDim*=2




def generatePretrainedVAEModel(projectModel,latentDim,noIters=1000, noSamplesPerIter = 8192):
    actionSpace = projectModel.getActionSpace()
    vae = Interface_VAE.VAE.VAE_Model(projectModel,latentDim,False)


    print("starting pre-training of VAE:")


    # get threshhold for good actions as mean minus standard deviation averaged with best of many random actions
    losses = [projectModel.simulate_returnLoss(actionSpace.sample()) for i in range(1000)]
    meanL = np.mean(losses)
    stdL = np.std(losses)
    bestL = np.min(losses)
    threshhold = np.mean([meanL - stdL, bestL])
    #threshhold = meanL-stdL
    print('threshhold: ' + str(threshhold))

    if True:
        # pre-train with best of following actions
        noSamplesTrainedOnTotal = 0
        for i in range(noIters):
            goodActions = [action for action in (actionSpace.sample() for i in range(noSamplesPerIter)) if projectModel.simulate_returnLoss(action) < threshhold]
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
    name = projectModel.modelOptions.asPretrainedVAE_Filename(latentDim)
    vae.model.save(name)

    #test if model can be loaded
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform

    model = load_model(name)
    print('loaded successfully')

'''generate and pre-train VAE'''
#test run
start = time.time()
generatePretrainedVAEModel(projectModel,latentDim,1,1 )
timeNeeded = time.time()-start
print("needed %f seconds" % timeNeeded)
#real run
start = time.time()
generatePretrainedVAEModel(projectModel,latentDim)
timeNeeded = time.time()-start
print("needed %f seconds" % timeNeeded)