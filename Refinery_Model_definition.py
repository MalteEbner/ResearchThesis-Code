import Refinery_Model_ProjectNetwork
import Refinery_Model_suppliers
import numpy as np


class Model_Refinery:
    def __init__(self):
        self.suppliers = Refinery_Model_suppliers.loadSupplierData()
        self.activities = Refinery_Model_ProjectNetwork.loadActivityData(self.suppliers)
        self.defaultLossFunction = lambda tupl: tupl[0] * 6000 + tupl[1]

    def getVariantNumbers(self):
        return [len(act.variants) for act in self.activities]

    def simulate(self,chosenVariantIndizes=[]):
        if chosenVariantIndizes == []:
            chosenVariantIndizes = [0 for i in range(len(self.activities))]
        for activity,index in zip(self.activities,chosenVariantIndizes):
            activity.variants[index].simulate()
        self.totalDuration = max(act.endpoint for act in self.activities)
        self.totalCost = sum([act.cost for act in self.activities])
        self.averageQuality = sum(act.quality * act.base_duration for act in self.activities)/sum(act.base_duration for act in self.activities)
        return self.totalDuration, self.totalCost, self.averageQuality

    def getStartpoint(self,lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        startIndizes = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            bestVariant = np.argmin(variantLosses) #get the best Variant w.r.t to the loss assuming it is a single-activity-project
            activity.variants[bestVariant].simulate() #assume all predecessors(and their quality) are optimal
            startIndizes.append(bestVariant)
        return startIndizes

    def getOptFunction(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        return lambda chosenVariantIndizes: lossFunction(self.simulate(chosenVariantIndizes))


    def __repr__(self):
        lines = ""
        for activity in self.activities:
            lines += "\n" + str(activity)
        return lines



model = Model_Refinery()
print("\nModel 1:")
print(model.simulate())
print(model)
print("\nModel 2:")
startpoint = model.getStartpoint()
print(model.simulate(startpoint))
print(model)





