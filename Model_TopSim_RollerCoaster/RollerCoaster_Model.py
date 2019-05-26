from Model_TopSim_RollerCoaster import RollerCoaster_LoadSchedule
from Model_TopSim_RollerCoaster import RollerCoaster_LossFunction
from Meta_Model import Meta_Model
from Meta_Model import commonFunctions
import mcerp3

import numpy as np


class Model_RollerCoaster(Meta_Model.MetaModel):
    def __init__(self,modelOptions):
        filename = '../Model_TopSim_RollerCoaster/RollerCoaster_Model_Definition.xlsx'
        activities = RollerCoaster_LoadSchedule.loadActivityData(filename,modelOptions)
        lossClass = RollerCoaster_LossFunction.RollerCoaster_Loss(filename)
        defaultLossFunction = lossClass.RollerCoaster_calcLoss
        #defaultLossFunction = lambda tupl: tupl[0]
        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction,modelOptions)
        #self.simulate = self.simulateStepwise_withEvents #the normal simulation is here the one with events

    def calcPerformanceFunction(self,activities):
        totalDuration = activities[-1].endpoint
        totalCost = sum([act.cost for act in activities])
        totalTechnology = sum([act.technology for act in activities])
        totalQuality = sum([act.quality for act in activities])
        return (totalDuration,totalCost,totalTechnology,totalQuality)

    def getGoodStartpoint(self):
        if self.modelOptions.probabilistic == False:
            return [1, 2, 3, 3, 3, 3, 0, 3, 0, 1, 3, 3, 3, 1, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 0, 3, 1, 2, 0]
        else:
            return self.getZeroStartpoint()

    def resetFunction(self):
        for act in self.activities:
            act.resetFunction()
        if not self.events == []:
            for event in self.events.values():
                event.resetFunction()

class Activity_RollerCoaster(Meta_Model.Activity):
    def __init__(self,modelOptions,variants,name,activityID):

        self.name = name
        self.predecessors =[]
        super().__init__([],variants,activityID)

class Variant_RollerCoaster(Meta_Model.Variant):
    def __init__(self,modelOptions,duration,cost,technology,quality,risk):
        self.risk = risk
        if modelOptions.probabilistic:
            factor = self.getPertFactor()
            self.duration = duration*factor
            self.cost = cost*factor
        else:
            self.duration = duration
            self.cost = cost
        self.technology = technology
        self.quality = quality
        super().__init__(self.simulate_RollerCoaster,self.simulateStep_RollerCoaster)

    def getPertFactor(self):
        risk = self.risk
        factor = commonFunctions.pertRV(1 / (1 + risk), 1, 1 + risk)
        return factor


    def simulate_RollerCoaster(self,model):
        self.ensureStartpoint()
        self.activity.endpoint = self.activity.startpoint + self.duration
        self.activity.cost = self.cost
        self.activity.technology = self.technology
        self.activity.quality = self.quality
        return self.duration, self.cost, self.technology, self.quality


    def simulateStep_RollerCoaster(self,model): #always probabilistic
        pass




'''
model = Model_RollerCoaster()
print("\nModel 1:")
print(model.simulate())
print(model)
print("\nModel 2:")
startpoint = model.getStartpoint()
print(model.simulate(startpoint))
print(model)
'''





