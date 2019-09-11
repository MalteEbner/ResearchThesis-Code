from Algo_ActorCritic.ActorCritic_Function import actorCritic_RunAlgo
from Interface.generateModel import generateModel
from Interface_VAE import VAE_interface
from Interface.Model_options import Model_options

withVAE = True

'''generate Model with its options'''
modelOptions = Model_options('Refinery')  # type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression = False
model = generateModel(modelOptions)
if withVAE:
    model = VAE_interface.VAE_Interface(model)


res = actorCritic_RunAlgo(model)