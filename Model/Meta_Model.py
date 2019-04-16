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






