
from Interface_VAE.VAE import VAE_Model
from Interface import ActionSpace, DefaultInterface
from gym import spaces
import numpy as np



class VAE_Interface(DefaultInterface.DefaultInterface):
    def __init__(self, projectModel,latentDim=16):
        self.projectModel = projectModel
        self.actionSpace = ActionSpace.ActionSpace([spaces.Box(-np.inf,np.inf,shape=(latentDim,)),])
        self.VAE = VAE_Model(projectModel.getActionSpace(),latentDim)


    def simulate(self,latentAction_s,kind='random',learningRateVAE = 0.1):
        if isinstance(latentAction_s,ActionSpace.Action):
            performances = self.simulate_onBatch([latentAction_s],kind,learningRateVAE)
        elif isinstance(latentAction_s,list):
            performances = self.simulate_onBatch(latentAction_s,kind,learningRateVAE)
        else:
            print('latentAction_s: ' + str(latentAction_s))
            raise NotImplementedError
        return performances

    def simulate_onBatch(self,latentActions,kind,learningRateVAE):
        #generate actions
        actions = self.VAE.latentActionToAction(latentActions,kind)
        #simulation of actions
        if isinstance(actions,ActionSpace.Action):
            actions = [actions]
        performances = [super().simulate(action) for action in actions]

        #update VAE
        self.VAE.update(actions,learningRateVAE)

        return performances



