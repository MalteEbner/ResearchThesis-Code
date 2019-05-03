from Model_Refinery import Refinery_Model_suppliers, Refinery_Model_ProjectNetwork
from Meta_Model import Meta_Model
import numpy as np


class Model_Refinery(Meta_Model.MetaModel):
    def __init__(self):
        filename = '/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/Master ETIT/10. Semester/Forschungsarbeit/Project management paper/Andre Heleno Project_management_business_game /SimGame translated - Andre Heleno.xlsx'
        self.suppliers = Refinery_Model_suppliers.loadSupplierData(filename)
        activities = Refinery_Model_ProjectNetwork.loadActivityData(self.suppliers,filename)
        defaultLossFunction = lambda tupl: tupl[0]*tupl[1]
        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction)

    def calcPerformanceFunction(self,activities):
        totalDuration = max(act.endpoint for act in activities)
        totalCost = sum([act.cost for act in activities])
        averageQuality = sum(act.quality * act.base_duration for act in activities) / sum(
            act.base_duration for act in activities)
        return (totalDuration,totalCost,averageQuality)


'''
Meta-Model for the refinery
'''

class Activity_refinery(Meta_Model.Activity):
    def __init__(self,predecessors,suppliers,type,base_duration,base_cost_per_day,name,activity_ID):

        self.name = name
        self.type = type
        self.base_duration = base_duration
        self.base_cost_per_day = base_cost_per_day
        self.activity_ID = activity_ID
        self.suppliers = suppliers

        simulateFunctions = [(lambda y: ( lambda: simulate_refinery(self, y)))(supplier) for supplier in suppliers]
        variants = [Meta_Model.Variant(self,simulateFunction) for simulateFunction in simulateFunctions]
        super().__init__(predecessors,variants)

    def __repr__(self):
        firstLine = "name: " + self.name + ", type: " +  self.type + ", base duration: " + str(self.base_duration) + ", base cost per day: " +  str(self.base_cost_per_day) + ", ID: " + str(self.activity_ID) #+ ", supplier: " + self.suppliers[self.variantID]
        secondLine = "duration: " + str(int(self.duration)) + ", cost: " +str(int(self.cost)) + ", quality: " + str(int(self.quality*100)) + "%, start: " + str(int(self.startpoint)) + ", end: " + str(self.endpoint)
        return secondLine + "  " + firstLine


class Supplier():
    def __init__(self,name,competences):
        self.name = name
        self.competences = competences

    def hasCompetenceType(self,type):
        hasCompType = type in self.competences
        return hasCompType

class Compentence():
    def __init__(self,durationEfficiency,costEfficiency,qualityEfficiency):
        self.durationEfficiency = durationEfficiency
        self.costEfficiency = costEfficiency
        self.qualityEfficiency = qualityEfficiency


def simulate_refinery(activity_refinery,supplier):
    competence = supplier.competences[activity_refinery.type]
    if len(activity_refinery.predecessors)>0:
        predecessorQualities = [pred.quality for pred in activity_refinery.predecessors]
        averagePredecessorQuality = np.mean(predecessorQualities)
        predecessorEndpoints = [pred.endpoint for pred in activity_refinery.predecessors]
        startpoint = np.max(predecessorEndpoints)
    else:
        averagePredecessorQuality = 1
        startpoint = int(0)
    quality = 0.75 * competence.qualityEfficiency + 0.25 * averagePredecessorQuality
    duration = int(np.ceil(activity_refinery.base_duration /(competence.durationEfficiency*quality**2)))
    cost = int(np.ceil(activity_refinery.base_cost_per_day* duration/competence.costEfficiency**1))
    endpoint = startpoint+duration
    activity_refinery.duration = duration
    activity_refinery.cost = cost
    activity_refinery.quality = quality
    activity_refinery.startpoint = startpoint
    activity_refinery.endpoint = endpoint
    return duration,cost,quality


'''
model = Model_Refinery()
print("\nModel 1:")
print(model.simulate())
print(model)
print("\nModel 2:")
startpoint = model.getStartpoint()
print(model.simulate(startpoint))
print(model)
'''




