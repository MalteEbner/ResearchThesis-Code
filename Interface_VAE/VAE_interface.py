
from Interface_VAE.VAE import VAE_Model
from Interface import ActionSpace, DefaultInterface
from gym import spaces
import numpy as np



class VAE_Interface(DefaultInterface.DefaultInterface):
    def __init__(self, projectModel,latentDim=8):
        self.projectModel = projectModel
        self.actionSpace = ActionSpace.ActionSpace([spaces.Box(-np.inf,np.inf,shape=(latentDim,)),])
        self.VAE = VAE_Model(projectModel.getActionSpace(),latentDim)
        self.baseline = 0
        self.baselineUpdateFactor = 0.01


    def simulate(self,latentAction_s,kind='random',learningRateVAE = 0.01):
        if isinstance(latentAction_s,ActionSpace.Action):
            performances = self.simulate_onBatch([latentAction_s],kind,learningRateVAE)[0]
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

        performances = [self._simulateOnProject(action) for action in actions]

        # update baseline
        losses = [perf[0] for perf in performances]
        meanLoss = np.mean(losses)
        self.baseline += self.baselineUpdateFactor*(meanLoss-self.baseline)

        # only train VAE on good actions
        goodActions = [actions[i] for i in range(len(losses)) if losses[i]<self. baseline]
        if len(goodActions)>0:
            self.VAE.update(goodActions,learningRateVAE)

        return performances

    def _simulateOnProject(self,action):
        return super().simulate(action)


    def simulate_returnLoss_onBatch(self,actions):
        return [performance[0] for performance in self.simulate(actions)]



