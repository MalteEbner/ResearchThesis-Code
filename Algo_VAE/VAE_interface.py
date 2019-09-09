from Interface.generateModel import generateModel
from Algo_VAE.VAE import VAE_Model


class VAE_Project_Model():
    def __init__(self,projectModel,latentDim=4):
        self.projectModel = projectModel
        self.actionSpace = projectModel.getActionSpace()
        self.VAE = VAE_Model(self.actionSpace,latentDim)

    def simulate(self,latentAction,kind='random',learningRateVAE = 0.1):
        losses = self.simulate_onBatch([latentAction],kind,learningRateVAE)
        return losses[0]

    def simulate_onBatch(self,latentActions,kind='random',learningRateVAE = 0.1):
        #generate actions
        actions = [self.VAE.latentActionToAction(latentAction,kind) for latentAction in latentActions]
        #simulation actions
        losses = [self.projectModel.simulate_returnLoss(action) for action in actions]
        #update VAE
        self.VAE.update(self.actions,learningRateVAE)

        return losses
