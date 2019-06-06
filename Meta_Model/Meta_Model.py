import numpy as np
from Meta_Model import Meta_Model_options
from Meta_Model import ActionSpace

import copy



class MetaModel:

    def __init__(self, activities, defaultLossFunction,calcPerformanceFunction, modelOptions, events = []):
        self.modelOptions = modelOptions
        self.events = events
        self.activities = activities # activities are a list
        self.defaultLossFunction = defaultLossFunction
        self.calcPerformanceFunction = calcPerformanceFunction
        if modelOptions.withEvents:
            self.simulate = self.simulateStepwise_withEvents  # the normal simulation is the one with events



    def orderedEventIDs(self):
        if self.events == []:
            return []
        eventIDs = list(self.events.keys())
        eventIDs.sort()
        return eventIDs

    def getActionSpace(self):
        variantNumsActivities = [len(act.variants) for act in self.activities]
        variantNumsEvents = [len(self.events[eventID].eventOptions) for eventID in self.orderedEventIDs()]
        actionSpace = ActionSpace.ActionSpace(variantNumsActivities,variantNumsEvents,self.modelOptions.withScheduleCompression)
        return actionSpace



    def simulate(self, action, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityIndizes = action.activityIndizes
        if self.modelOptions.withScheduleCompression:
            for activity, index, compression in zip(self.activities, activityIndizes, action.scheduleCompressionFactors):
                activity.variants[index].simulate(self,compression)
        else:
            for activity, index in zip(self.activities, activityIndizes):
                activity.variants[index].simulate(self)

        self.performance = self.calcPerformanceFunction(self.activities)
        self.loss = lossFunction(self.performance)
        retValue = (self.loss,) + self.performance

        self.resetFunction()
        return retValue


    def resetFunction(self):
        for act in self.activities:
            act.resetFunction()
        pass



    def simulateMean(self,action, lossFunction=[], randomTestsToMean=[]):
        if randomTestsToMean==[]:
            randomTestsToMean=20
        performances = [self.simulate(action,lossFunction) for i in range(randomTestsToMean)]
        meanPerformance = np.mean(performances,axis=0)
        meanPerformance = np.round(meanPerformance,2)
        return meanPerformance

    def simulate_returnLoss(self, action, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        perf = self.simulate(action,lossFunction)
        loss = perf[0]
        return loss

    def simulateMean_returnLoss(self,action, lossFunction=[], randomTestsToMean=[]):
        loss = self.simulateMean(action,lossFunction,randomTestsToMean)[0]
        return loss

    def getGoodStartpoint(self, actionSpace,lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        startIndizes_activities = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate(self)) for var in activity.variants]
            bestVariant = np.argmin(
                variantLosses)  # get the best Variant w.r.t to the loss assuming it is a single-activity-project
            activity.variants[bestVariant].simulate(self)  # assume all predecessors(and their quality) are optimal
            startIndizes_activities.append(bestVariant)

        action = ActionSpace.Action(actionSpace)
        action.saveDirectly(startIndizes_activities,[0 for i in self.events])
        return action


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



    def simulateStepwise_withEvents(self,action,lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityIndizes = action.activityIndizes
        eventOptionDict = action.eventIndizesAsDict(self.orderedEventIDs())

        chosenVariantIndizes = action.activityIndizes

        self.Time = 0
        self.TimeDelay = 0
        self.noEventsPlayed = 0

        activities = self.activities
        events = self.events

        #while last activity has not finished yet
        while not activities[-1].finished:

            if self.TimeDelay == 0:
                #do one step on all variants
                for activity,chosenVariantIndex in zip(activities,chosenVariantIndizes):
                    if all(act.finished for act in activity.predecessors):
                        activity.variants[chosenVariantIndex].simulateStep(self)
            else:
                self.TimeDelay -=1


            # run all occuring events
            for eventID in events.keys():
                event = events[eventID]
                eventOptionIndex = eventOptionDict[eventID]
                event.runEvent(self,eventOptionIndex)

            #increase time
            self.Time += 1
            #print(self.Time)

        self.performance = self.calcPerformanceFunction(activities)
        self.loss = lossFunction(self.performance)
        retValue = (self.loss,) + self.performance
        self.resetFunction()
        return retValue

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
        if hasattr(self, 'startpoint'):
            del self.startpoint
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

    def ensureStartpoint(self):
        if not hasattr(self.activity,'startpoint'):
            if len(self.activity.predecessors) > 0:
                self.activity.startpoint = max(pred.endpoint for pred in self.activity.predecessors)
            else:
                self.activity.startpoint = int(0)
            self.progress = 0

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







