import numpy as np
from Meta_Model import Meta_Model



class MetaModel_stepwise(Meta_Model.MetaModel):
    def __init__(self,activities, defaultLossFunction,calcPerformanceFunction,events=[]):
        self.events = events  # events are a dictionary with eventID as key
        self.noEventsPlayed = 0
        Meta_Model.MetaModel.__init__(self,activities, defaultLossFunction,calcPerformanceFunction)

    def orderedEventIDs(self):
        if self.events ==[]:
            return []
        eventIDs = list(self.events.keys())
        eventIDs.sort()
        return eventIDs

    def getVariantNumbers(self):
        variantNumsActivities = [len(act.variants) for act in self.activities]
        variantNumsEvents = [len(self.events[eventID].eventOptions) for eventID in self.orderedEventIDs()]
        return variantNumsActivities + variantNumsEvents

    def chosenVariantIndizes_activititesEvents_to_seperateOnes(self,chosenVariantIndizes_activitiesEvents):
        noActivities = len(self.activities)
        chosenActivityVariantIndizes = chosenVariantIndizes_activitiesEvents[:noActivities]
        chosenEventOptionIndizes_List = chosenVariantIndizes_activitiesEvents[noActivities:]
        eventIDs = self.orderedEventIDs()
        chosenEventOptionIndizes = dict(zip(eventIDs,chosenEventOptionIndizes_List))
        return chosenActivityVariantIndizes,chosenEventOptionIndizes

    def simulateStepwise_withEvents(self,chosenVariantIndizes_activitiesEvents,lossFunction=[]):
        if lossFunction == []:
            lossFunction = self.defaultLossFunction

        chosenVariantIndizes, chosenEventOptionIndizes = self.chosenVariantIndizes_activititesEvents_to_seperateOnes(chosenVariantIndizes_activitiesEvents)

        self.Time = 0
        self.TimeDelay = 0
        self.noEventsPlayed = 0

        #activities = copy.deepcopy(self.activities)
        #events = copy.deepcopy(self.events)
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
                eventOptionIndex = chosenEventOptionIndizes[eventID]
                event.runEvent(self,eventOptionIndex)

            #increase time
            self.Time += 1
            #print(self.Time)

        self.performance = self.calcPerformanceFunction(activities)
        self.loss = lossFunction(self.performance)
        retValue = (self.loss,) + self.performance
        self.resetFunction()
        return retValue
