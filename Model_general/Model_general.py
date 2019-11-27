import numpy as np
from Interface import ActionSpace
from Interface import ProjectActionSpace


class Model_general:

    def __init__(self, activities, defaultLossFunction, calcPerformanceFunction, modelOptions, events = []):
        self.modelOptions = modelOptions
        self.events = events
        self.activities = activities # activities are a list
        self.defaultLossFunction = defaultLossFunction
        self.calcPerformanceFunction = calcPerformanceFunction
        self.modelOptions = modelOptions
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
        actionSpace = ProjectActionSpace.ProjectActionSpace(variantNumsActivities, variantNumsEvents, self.modelOptions.withScheduleCompression)
        return actionSpace



    def simulate(self, action, lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityIndizes = action.activityIndizes()
        if self.modelOptions.withScheduleCompression:
            for activity, index, compression in zip(self.activities, activityIndizes, action.scheduleCompressionFactors()):
                if index >= len(activity.variants):
                    I=0
                activity.variants[index].simulate(compression)
        else:
            for activity, index in zip(self.activities, activityIndizes):
                activity.variants[index].simulate()

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
            randomTestsToMean=10
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

    def getHeuristicBestAction(self,lossFunction=[]):
        action = ProjectActionSpace.ProjectAction(self.getActionSpace())
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        bestActivities = []
        for activity in self.activities:
            variantLosses = [lossFunction(var.simulate()) for var in activity.variants]
            bestLoss = np.argmin(variantLosses)
            bestActivities.append(bestLoss)
        bestEvents = [0 for i in self.orderedEventIDs()]
        valuesList = [bestActivities,bestEvents]
        if self.modelOptions.withScheduleCompression:
            scheduleCompressionFactors = [1 for i in self.activities]
            valuesList.append(scheduleCompressionFactors)
        action.saveValuesList(valuesList)
        self.resetFunction()
        return action

    def getZeroAction(self):
        action = ProjectActionSpace.ProjectAction(self.getActionSpace())
        bestActivities = [0 for i in self.activities]
        bestEvents = [0 for i in self.orderedEventIDs()]
        valuesList = [bestActivities,bestEvents]
        if self.modelOptions.withScheduleCompression:
            scheduleCompressionFactors = [1 for i in self.activities]
            valuesList.append(scheduleCompressionFactors)
        action.saveValuesList(valuesList)
        self.resetFunction()
        return action




    def simulateStepwise_withEvents(self,action,lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction
        activityIndizes = action.activityIndizes()
        eventOptionDict = action.eventIndizesAsDict(self.orderedEventIDs())


        self.Time = 0
        self.TimeDelay = 0
        self.noEventsPlayed = 0

        activities = self.activities
        events = self.events

        #while last activity has not finished yet
        while not activities[-1].finished:

            if self.TimeDelay == 0:
                #do one step on all variants
                if self.modelOptions.withScheduleCompression:
                    for activity, chosenVariantIndex, compression in zip(self.activities, activityIndizes, action.scheduleCompressionFactors()):
                        if all(act.finished for act in activity.predecessors):
                            activity.variants[chosenVariantIndex].simulateStep(compression)
                else:
                    for activity, index in zip(self.activities, activityIndizes):
                        activity.variants[index].simulateStep()
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

    def noOptions(self):
        variantNumsActivities = [len(act.variants) for act in self.activities]
        min_ = min(variantNumsActivities)
        max_ = max(variantNumsActivities)
        len_act=len(variantNumsActivities)
        variantNumsEvents = [len(self.events[eventID].eventOptions) for eventID in self.orderedEventIDs()]
        len_events = len(variantNumsEvents)
        if len_events>0:
            min_e = min(variantNumsEvents)
            max_e = max(variantNumsEvents)
        variantNums = [float(i) for i in variantNumsActivities+variantNumsEvents if i>0]
        noOptions = np.product(variantNums)
        return noOptions


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

    def simulateStep(self,scheduleCompressionFactor=1):
        if all(pred.finished for pred in self.activity.predecessors):
            relProgress = self.simulateStepFunction(scheduleCompressionFactor)
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







