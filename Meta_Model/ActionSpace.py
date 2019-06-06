import random

class ActionSpace:
    def __init__(self,activityVariantNumbers,eventVariantNumbers=[],withScheduleCompression=False):
        self.withScheduleCompression = withScheduleCompression
        self.noActivities = len(activityVariantNumbers)
        self.noEvents = len(eventVariantNumbers)
        self.activityVariantNumbers = activityVariantNumbers
        self.eventVariantNumbers = eventVariantNumbers

    def VariantNumbers(self):
        return self.activityVariantNumbers + self.eventVariantNumbers

    def getZeroAction(self):
        action = Action(self)
        startIndizes_activities = [int(0) for i in self.activityVariantNumbers]
        startIndizes_events = [int(0) for i in self.eventVariantNumbers]
        if self.withScheduleCompression:
            startScheduleCompressionFactors = [1.0 for i in self.activityVariantNumbers]
        else:
            startScheduleCompressionFactors = []
        action.saveDirectly(startIndizes_activities,startIndizes_events,startScheduleCompressionFactors)
        return action

    def getRandomAction(self):
        action = Action(self)
        startIndizes_activities = [random.randrange(i) for i in self.activityVariantNumbers]
        startIndizes_events = [random.randrange(i) for i in self.eventVariantNumbers]
        if self.withScheduleCompression:
            startScheduleCompressionFactors = [random.uniform(0.5,1) for i in self.activityVariantNumbers]
        else:
            startScheduleCompressionFactors = []
        action.saveDirectly(startIndizes_activities, startIndizes_events, startScheduleCompressionFactors)
        return action



class Action:
    def __init__(self,actionSpace):
        self.actionSpace = actionSpace


    def saveDirectly(self,chosenVariantIndizes_activities,chosenVariantIndizes_events=[],scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes_activities
        self.eventIndizes = chosenVariantIndizes_events
        self.scheduleCompressionFactors = scheduleCompressionFactors


    def saveIndizesCombined(self,chosenVariantIndizes,scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes[:self.actionSpace.noActivities]
        self.eventIndizes = chosenVariantIndizes[self.actionSpace.noActivities:]
        self.scheduleCompressionFactors = scheduleCompressionFactors


    def eventIndizesAsDict(self,orderedEventIDs):
        eventIndizesDict = dict(zip(orderedEventIDs,self.eventIndizes))
        return eventIndizesDict

    def __repr__(self):
        lines = "activities:"
        lines += str(self.activityIndizes)
        if self.actionSpace.noEvents>0:
            lines += '\n events:'
            lines += str(self.eventIndizes)
        if self.actionSpace.withScheduleCompression:
            lines += '\n compressionFactors:'
            lines += str(self.scheduleCompressionFactors)
        return lines

