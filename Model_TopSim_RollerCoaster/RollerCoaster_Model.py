from Model_TopSim_RollerCoaster import RollerCoaster_LoadSchedule
from Model_TopSim_RollerCoaster import RollerCoaster_LossFunction
from Meta_Model import Meta_Model
from Meta_Model import Meta_Model_stepwise
from Meta_Model import commonFunctions
import mcerp3

import numpy as np


class Model_RollerCoaster(Meta_Model_stepwise.MetaModel_stepwise):
    def __init__(self,probabilisticModel=True):
        filename = '/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/Master ETIT/10. Semester/Forschungsarbeit/Code Forschungsarbeit/Model_TopSim_RollerCoaster/RollerCoaster_Model_Definition.xlsx'
        activities = RollerCoaster_LoadSchedule.loadActivityData(filename,probabilisticModel)
        lossClass = RollerCoaster_LossFunction.RollerCoaster_Loss(filename)
        defaultLossFunction = lossClass.RollerCoaster_calcLoss
        #defaultLossFunction = lambda tupl: tupl[0]
        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction)
        self.simulate = self.simulateStepwise_withEvents #the normal simulation is here the one with events

    def calcPerformanceFunction(self,activities):
        totalDuration = activities[-1].endpoint
        totalCost = sum([act.cost for act in activities])
        totalTechnology = sum([act.technology for act in activities])
        totalQuality = sum([act.quality for act in activities])
        return (totalDuration,totalCost,totalTechnology,totalQuality)

    def getGoodStartpoint(self):
        return [1, 2, 2, 3, 3, 3, 0, 3, 2, 1, 0, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 0, 3, 1, 2, 0]

class Activity_RollerCoaster(Meta_Model.Activity):
    def __init__(self,variants,name,activityID,probabilisticModel):

        self.name = name
        self.predecessors =[]
        super().__init__([],variants,activityID)

class Variant_RollerCoaster(Meta_Model.Variant):
    def __init__(self,duration,cost,technology,quality,risk):
        factor = self.getPertFactor()
        self.neededDuration = duration * factor
        self.baseCost = cost
        self.technology = technology
        self.quality = quality
        self.risk = risk
        super().__init__(self,self.simulate_RollerCoaster,self.simulateStep_RollerCoaster)

    def getPertFactor(self):
        risk = self.risk
        factor = commonFunctions.pertRV(1 / (1 + risk), 1, 1 + risk)
        return factor


    def simulate_RollerCoaster(self,probabilisticModel):
        activity = self.activity
        if len(activity.predecessors)>0:
            startpoint = max(pred.endpoint for pred in activity.predecessors)
        else:
            startpoint = int(0)
        duration = self.duration
        cost = self.cost
        if probabilisticModel:
            factor = self.getPertFactor()
            duration *= factor
            cost *= factor
        activity.startpoint = startpoint
        activity.endpoint = startpoint + duration
        activity.cost = self.cost
        activity.technology = self.technology
        activity.quality = self.quality
        return self.duration, self.cost, self.technology, self.quality

    def simulateStep_RollerCoaster(self): #always probabilistic




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





