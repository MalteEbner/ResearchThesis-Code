from Algo_ActorCritic.ActorCritic_Function import actorCritic_RunAlgo
from Interface.generateModel import generateModel
from Interface_VAE import VAE_interface
from Interface.Model_options import Model_options



'''generate Model with its options'''
modelOptions = Model_options('MIS') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
#modelOptions.probabilistic = True
modelOptions.withScheduleCompression=True
#modelOptions.interface='VAE'
model = generateModel(modelOptions)


res = actorCritic_RunAlgo(model)