import numpy as np

import copy


class MetaModel:

    def __init__(self, activities, defaultLossFunction,calcPerformanceFunction):
        self.activities = activities # activities are a list
        self.defaultLossFunction = defaultLossFunction
        self.calcPerformanceFunction = calcPerformanceFunction


    def getVariantNumbers(self):
        variantNumsActivities = [len(act.variants) for act in self.activities]
        return variantNumsActivities

    def simulate(self, chosenVariantIndizes, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        for activity, index in zip(self.activities, chosenVariantIndizes):
            activity.variants[index].simulate()

        self.performance = self.calcPerformanceFunction(self.activities)
        self.loss = lossFunction(self.performance)
        retValue = (self.loss,) + self.performance
        return retValue


    def resetFunction(self):
        pass



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
        startIndizes_activities = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            bestVariant = np.argmin(
                variantLosses)  # get the best Variant w.r.t to the loss assuming it is a single-activity-project
            activity.variants[bestVariant].simulate()  # assume all predecessors(and their quality) are optimal
            startIndizes_activities.append(bestVariant)
        return startIndizes_activities + self.getZeroStartpoint_events()

    def getZeroStartpoint(self):
        startIndizes_activities = [int(0) for act in self.activities]
        return startIndizes_activities + self.getZeroStartpoint_events()

    def getZeroStartpoint_events(self):
        startIndizes = [0 for i in self.events]
        return startIndizes

    def getActivityLosses(self, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        losses = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            losses.append(variantLosses)
        for eventID in self.orderedEventIDs():
            optionsLosses = [0 for y in self.events[eventID].eventOptions]
            losses.append(optionsLosses)
        return losses

    def __repr__(self):
        lines = ""
        for activity in self.activities:
            lines += "\n" + str(activity)
        return lines

class Activity():
    def __init__(self, predecessors, variants, activityID):
        self.predecessors = predecessors
        self.variants = variants
        self.activityID = activityID
        self.finished = False
        for var in variants:
            var.activity = self

    def resetFunction(self):
        self.finished = False
        for var in self.variants:
            var.resetFunction()


class Variant():
    def __init__(self, simulate, simulateStepFunction=[]):
        self.simulateStepFunction = simulateStepFunction
        self.simulate = simulate

    def simulateStep(self,model):
        if all(pred.finished for pred in self.activity.predecessors):
            relProgress = self.simulateStepFunction(model)
            if relProgress >= 1:
                self.activity.finished = True

    def resetFunction(self):
        pass

class Event():
    def __init__(self, occurCondition, eventOptions, onlyOnce=True):
        self.allowRun = True
        self.onlyOnce = onlyOnce
        self.occurCondition = occurCondition
        self.eventOptions = eventOptions

    def runEvent(self,meta_model,eventOptionIndex):
        if self.allowRun and self.occurCondition(meta_model):
            self.eventOptions[eventOptionIndex].RunEventOption(meta_model)
            #print("")
            #print("time: "+ str(meta_model.Time))
            #print("ran event "+ str(self.eventID) + ": " + self.description)
            #print("chose option " + str(eventOptionIndex+1) + ": " + self.eventOptions[eventOptionIndex].description)
            meta_model.noEventsPlayed +=1


            #only allow running it once if necessary
            if self.onlyOnce==True:
                self.allowRun = False

    def resetFunction(self):
        self.allowRun = True
        for eventOption in self.eventOptions:
            eventOption.resetFunction()

class EventOption():
    def __init__(self,runFunction):
        self.RunEventOption = runFunction

    def resetFunction(self):
        pass







