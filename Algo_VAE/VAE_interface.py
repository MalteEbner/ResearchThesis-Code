from Interface.generateModel import generateModel
from Algo_VAE.VAE import VAE_Model
from Interface import ActionSpace
from gym import spaces
import numpy as np
from Algo_ActorCritic import ActorCritic_general


class VAE_Project_Model():
    def __init__(self, projectModel,latentDim=4):
        self.projectModel = projectModel
        self.actionSpace = spaces.Tuple((spaces.Box(-np.inf,np.inf,shape=(latentDim,)),))
        self.VAE = VAE_Model(projectModel.getActionSpace(),latentDim)

    def getActionSpace(self):
        return self.actionSpace

    def simulate(self,latentAction_s,kind='random',learningRateVAE = 0.1):
        if isinstance(latentAction_s,tuple):
            latentAction = np.expand_dims(latentAction_s[0],axis=0)
            losses = self.simulate_onBatch(latentAction,kind,learningRateVAE)
        elif isinstance(latentAction_s,list):
            losses = self.simulate_onBatch(latentAction_s,kind,learningRateVAE)
        else:
            print('latentAction_s: ' + str(latentAction_s))
            raise NotImplementedError
        return losses[0]

    def simulate_onBatch(self,latentActions,kind,learningRateVAE):
        #generate actions
        actions = self.VAE.latentActionToAction(latentActions,kind)
        #simulation of actions
        if isinstance(actions,ActionSpace.Action):
            actions = [actions]
        losses = [self.projectModel.simulate_returnLoss(action) for action in actions]

        #update VAE
        self.VAE.update(actions,learningRateVAE)

        return losses

    def simulate_returnLoss(self,latentAction):
        return self.simulate(latentAction)

    def sampleAction(self):
        return self.actionSpace.sample()

