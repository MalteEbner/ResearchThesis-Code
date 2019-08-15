import random
from keras.utils import to_categorical
from keras.backend import expand_dims

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

    def saveEverythingCombined(self,completeInput):
        noCategoricalVariables = self.actionSpace.noActivities+self.actionSpace.noEvents
        self.activityIndizes = completeInput[:self.actionSpace.noActivities].astype('int')
        self.eventIndizes = completeInput[self.actionSpace.noActivities:noCategoricalVariables].astype('int')
        self.scheduleCompressionFactors = completeInput[noCategoricalVariables:]

    def saveIndizesCombined(self,chosenVariantIndizes,scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes[:self.actionSpace.noActivities].astype('int')
        self.eventIndizes = chosenVariantIndizes[self.actionSpace.noActivities:].astype('int')
        self.scheduleCompressionFactors = scheduleCompressionFactors


    def getOneHotCoded(self):
        action = []
        for index, noVariants in zip(self.activityIndizes,self.actionSpace.activityVariantNumbers):
            encoding = to_categorical(index,num_classes=noVariants)
            encoding = expand_dims(encoding,axis=0)
            encoding = expand_dims(encoding, axis=0)
            action.append(encoding)
        for index, noVariants in zip(self.eventIndizes,self.actionSpace.eventVariantNumbers):
            encoding = to_categorical(index,num_classes=noVariants)
            encoding = expand_dims(encoding, axis=0)
            encoding = expand_dims(encoding, axis=0)
            action.append(encoding)
        scheduleCompressionFactors = [[i] for i in self.scheduleCompressionFactors]
        action += list(self.scheduleCompressionFactors)
        return action


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

