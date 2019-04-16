import numpy as np


class MetaModel:

    def __init__(self, activities, defaultLossFunction):
        self.activities = activities
        self.defaultLossFunction = defaultLossFunction

    def getVariantNumbers(self):
        return [len(act.variants) for act in self.activities]

    def simulate(self, chosenVariantIndizes=[]):
        if chosenVariantIndizes == []:
            chosenVariantIndizes = [0 for i in range(len(self.activities))]
        for activity, index in zip(self.activities, chosenVariantIndizes):
            activity.variants[index].simulate()
        self.totalDuration = max(act.endpoint for act in self.activities)
        self.totalCost = sum([act.cost for act in self.activities])
        self.averageQuality = sum(act.quality * act.base_duration for act in self.activities) / sum(
            act.base_duration for act in self.activities)
        return self.totalDuration, self.totalCost, self.averageQuality

    def simulate_returnLoss(self, chosenVariantIndizes, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        loss = lossFunction(self.simulate(chosenVariantIndizes))
        return loss

    def getStartpoint(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        startIndizes = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            bestVariant = np.argmin(
                variantLosses)  # get the best Variant w.r.t to the loss assuming it is a single-activity-project
            activity.variants[bestVariant].simulate()  # assume all predecessors(and their quality) are optimal
            startIndizes.append(bestVariant)
        return startIndizes

    def getActivityLosses(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityLosses = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            activityLosses.append(variantLosses)
        return activityLosses

    def getOptFunction(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        return lambda chosenVariantIndizes: lossFunction(self.simulate(chosenVariantIndizes))

    def __repr__(self):
        lines = ""
        for activity in self.activities:
            lines += "\n" + str(activity)
        return lines

'''
general Meta-Model
'''

class Activity():
    def __init__(self, predecessors, variants):
        self.predecessors = predecessors
        self.variants = variants

    def chooseVariant(self, variantIndex):
        self.variantID = variantIndex
        self.chosenVariant = self.variants[variantIndex]




class Variant():
    def __init__(self, simulate, simulateStep=[]):
        self.simulate = simulate
        self.simulateStep = simulateStep



'''
Meta-Model for the refinery
'''

class Activity_refinery(Activity):
    def __init__(self,predecessors,suppliers,type,base_duration,base_cost_per_day,name,activity_ID):

        self.name = name
        self.type = type
        self.base_duration = base_duration
        self.base_cost_per_day = base_cost_per_day
        self.activity_ID = activity_ID
        self.suppliers = suppliers

        simulateFunctions = [(lambda y: ( lambda: simulate_refinery(self, y)))(supplier) for supplier in suppliers]
        variants = [Variant(simulateFunction) for simulateFunction in simulateFunctions]
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


