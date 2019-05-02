import numpy as np


class MetaModel:

    def __init__(self, activities, defaultLossFunction,calcPerformanceFunction):
        self.activities = activities
        self.defaultLossFunction = defaultLossFunction
        self.calcPerformanceFunction = calcPerformanceFunction

    def getVariantNumbers(self):
        return [len(act.variants) for act in self.activities]

    def simulate(self, chosenVariantIndizes, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        for activity, index in zip(self.activities, chosenVariantIndizes):
            activity.variants[index].simulate()
        self.performance = self.calcPerformanceFunction(self.activities)
        self.loss = lossFunction(self.performance)
        retValue = (self.loss,) + self.performance
        return retValue

    def simulateMean(self,chosenVariantIndizes, lossFunction=[], randomTestsToMean=[]):
        if randomTestsToMean==[]:
            randomTestsToMean=20
        noTests = int(randomTestsToMean)
        performances = [self.simulate(chosenVariantIndizes,lossFunction) for i in range(randomTestsToMean)]
        meanPerformance = np.mean(performances,axis=0)
        meanPerformance = np.round(meanPerformance,2)
        return meanPerformance

    def simulate_returnLoss(self, chosenVariantIndizes, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        loss = self.simulate(chosenVariantIndizes,lossFunction)[0]
        return loss

    def simulateMean_returnLoss(self,chosenVariantIndizes, lossFunction=[], randomTestsToMean=[]):
        loss = self.simulateMean(chosenVariantIndizes,lossFunction,randomTestsToMean)[0]
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

    def getZeroStartpoint(self):
        startIndizes = [int(0) for act in self.activities]
        return startIndizes

    def getActivityLosses(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityLosses = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            activityLosses.append(variantLosses)
        return activityLosses

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






