from Model_TopSim_RollerCoaster import RollerCoaster_LoadSchedule
from Model_TopSim_RollerCoaster import RollerCoaster_LossFunction
from Meta_Model import Meta_Model
import random
import numpy as np


class Model_RollerCoaster(Meta_Model.MetaModel):
    def __init__(self,probabilisticModel=True):
        filename = '/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/Master ETIT/10. Semester/Forschungsarbeit/Code Forschungsarbeit/Model_TopSim_RollerCoaster/RollerCoaster_Model_Definition.xlsx'
        activities = RollerCoaster_LoadSchedule.loadActivityData(filename,probabilisticModel)
        lossClass = RollerCoaster_LossFunction.RollerCoaster_Loss(filename)
        defaultLossFunction = lossClass.RollerCoaster_calcLoss
        #defaultLossFunction = lambda tupl: tupl[0]
        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction)

    def calcPerformanceFunction(self,activities):
        totalDuration = activities[-1].endpoint
        totalCost = sum([act.cost for act in activities])
        totalTechnology = sum([act.technology for act in activities])
        totalQuality = sum([act.quality for act in activities])
        return (totalDuration,totalCost,totalTechnology,totalQuality)

    def getGoodStartpoint(self):
        return [1, 2, 2, 3, 3, 3, 0, 3, 2, 1, 0, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 0, 3, 1, 2, 0]

class Activity_RollerCoaster(Meta_Model.Activity):
    def __init__(self,variantDataList,name,activityID,probabilisticModel):

        self.name = name
        self.predecessors =[]
        simulateFunctions = simulateFunctions = [(lambda y: ( lambda: simulate_RollerCoaster(self, y,probabilisticModel)))(variantData) for variantData in variantDataList]
        variants = [Meta_Model.Variant(simulateFunction) for simulateFunction in simulateFunctions]
        super().__init__([],variants,activityID)

class VariantData_RollerCoaster():
    def __init__(self,duration,cost,technology,quality,risk):
        self.duration = duration
        self.cost = cost
        self.technology = technology
        self.quality = quality
        self.risk = risk


def simulate_RollerCoaster(activity,variantData,probabilisticModel):
    if len(activity.predecessors)>0:
        startpoint = max(pred.endpoint for pred in activity.predecessors)
    else:
        startpoint = int(0)
    duration = variantData.duration
    cost = variantData.cost
    if probabilisticModel and random.random() < variantData.risk:
        duration *= 1 + variantData.risk
        cost *= 1 + variantData.risk
    activity.startpoint = startpoint
    activity.endpoint = startpoint + duration
    activity.cost = variantData.cost
    activity.technology = variantData.technology
    activity.quality = variantData.quality
    return variantData.duration, variantData.cost, variantData.technology, variantData.quality


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





