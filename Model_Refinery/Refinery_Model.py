from Model_Refinery import Refinery_Model_suppliers, Refinery_Model_ProjectNetwork
from Meta_Model import Meta_Model
import numpy as np


class Model_Refinery(Meta_Model.MetaModel):
    def __init__(self,modelOptions):
        filename = '../Model_Refinery/SimGame translated - Andre Heleno.xlsx'
        self.suppliers = Refinery_Model_suppliers.loadSupplierData(filename)
        activities = Refinery_Model_ProjectNetwork.loadActivityData(self.suppliers,filename)
        defaultLossFunction = lambda tupl: tupl[0]*tupl[1]
        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction,modelOptions)

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
        self.suppliers = suppliers

        variants = [Variant_Refinery(supplier) for supplier in suppliers]
        super().__init__(predecessors,variants,activity_ID)

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

class Variant_Refinery(Meta_Model.Variant):
    def __init__(self,supplier):
        self.supplier = supplier

    def simulate(self,model):
        self.ensureStartpoint()

        if len(self.activity.predecessors)>0:
            predecessorQualities = [pred.quality for pred in self.activity.predecessors]
            averagePredecessorQuality = np.mean(predecessorQualities)
        else:
            averagePredecessorQuality = 1

        competence = self.supplier.competences[self.activity.type]
        quality = 0.75 * competence.qualityEfficiency + 0.25 * averagePredecessorQuality
        duration = int(np.ceil(self.activity.base_duration /(competence.durationEfficiency*quality**2)))
        cost = int(np.ceil(self.activity.base_cost_per_day* duration/competence.costEfficiency**1))
        endpoint = self.activity.startpoint+duration

        self.activity.duration = duration
        self.activity.cost = cost
        self.activity.quality = quality
        self.activity.endpoint = endpoint

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




