import random
from keras.utils import to_categorical
from keras.backend import expand_dims
from gym import spaces

class ActionSpace(spaces.Tuple):
    def __init__(self,activityVariantNumbers,eventVariantNumbers=[],withScheduleCompression=False):
        self.noActivities = len(activityVariantNumbers)
        self.noEvents = len(eventVariantNumbers)
        self.withScheduleCompression = withScheduleCompression

        #define Action Spaces using the 'spaces' class provided by the openAI gym
        spaceDict = {}
        activityVariantSpace = spaces.MultiDiscrete(activityVariantNumbers)
        eventVariantSpace = spaces.MultiDiscrete(eventVariantNumbers)
        space = (activityVariantSpace,eventVariantSpace)
        if self.withScheduleCompression:
            scheduleCompressionSpace = spaces.Box(0.5,1,shape=(self.noActivities,))
            space = space + (scheduleCompressionSpace,)
        super().__init__(spaces=space)


    def NoAllVariables(self):
        noVars = self.noActivities+self.noEvents
        if self.withScheduleCompression:
            noVars += self.noActivities
        return noVars

    def VariantNumbers(self):
        varNumbers = self.ActivityVariantNumbers() + self.EventVariantNumbers()
        return varNumbers

    def ActivityVariantNumbers(self):
        return list(self.spaces[0].nvec)

    def EventVariantNumbers(self):
        return list(self.spaces[1].nvec)

    def sampleZeroAction(self):
        action = Action(self)
        startIndizes = [int(0) for i in self.VariantNumbers()]
        if self.withScheduleCompression:
            startScheduleCompressionFactors = [1.0 for i in self.ActivityVariantNumbers()]
        else:
            startScheduleCompressionFactors = []
        action.saveIndizesCombined(startIndizes,startScheduleCompressionFactors)
        return action

    def sampleAction(self):
        samples = self.sample()
        action = Action(self)
        action.saveWithTuple(samples)
        return action

    def sample(self):
        return self.sampleAction()




class Action:
    def __init__(self,actionSpace):
        self.actionSpace = actionSpace

    def saveWithTuple(self,samples):
        if self.actionSpace.withScheduleCompression:
            self.saveDirectly(samples[0],samples[1],samples[2])
        else:
            self.saveDirectly(samples[0],samples[1])

    def saveDirectly(self,chosenVariantIndizes_activities,chosenVariantIndizes_events=[],scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes_activities
        self.eventIndizes = chosenVariantIndizes_events
        self.scheduleCompressionFactors = scheduleCompressionFactors

    def saveEverythingCombined(self,completeInput):
        noCategoricalVariables = self.actionSpace.noActivities+self.actionSpace.noEvents
        self.activityIndizes = [int(i) for i in completeInput[:self.actionSpace.noActivities]]
        self.eventIndizes = [int(i) for i in completeInput[self.actionSpace.noActivities:noCategoricalVariables]]
        self.scheduleCompressionFactors = completeInput[noCategoricalVariables:]

    def saveIndizesCombined(self,chosenVariantIndizes,scheduleCompressionFactors=[]):
        self.activityIndizes = [int(i) for i in chosenVariantIndizes[:self.actionSpace.noActivities]]
        self.eventIndizes = [int(i) for i in chosenVariantIndizes[self.actionSpace.noActivities:]]
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

