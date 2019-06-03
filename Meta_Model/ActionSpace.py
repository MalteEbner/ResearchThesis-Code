

class ActionSpace:
    def __init__(self,noActivities,noEvents=0):
        self.noActivities = noActivities
        self.noEvents = noEvents


    def saveDirectly(self,chosenVariantIndizes_activities,chosenVariantIndizes_events=[],scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes_activities
        self.eventIndizes = chosenVariantIndizes_events
        self.compressionFactors = scheduleCompressionFactors


    def saveIndizesCombined(self,chosenVariantIndizes,scheduleCompressionFactors=[]):
        self.activityIndizes = chosenVariantIndizes[:self.noActivities]
        self.eventIndizes = chosenVariantIndizes[self.noActivities:]
        self.scheduleCompressionFactors = scheduleCompressionFactors
