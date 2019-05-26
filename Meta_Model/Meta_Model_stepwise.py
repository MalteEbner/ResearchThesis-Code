import numpy as np
from Meta_Model import Meta_Model



class asdfMetaModel_stepwise(Meta_Model.MetaModel):
    def __init__(self,activities, defaultLossFunction,calcPerformanceFunction,events=[]):
        self.events = events  # events are a dictionary with eventID as key
        self.noEventsPlayed = 0
        Meta_Model.MetaModel.__init__(self,activities, defaultLossFunction,calcPerformanceFunction)


