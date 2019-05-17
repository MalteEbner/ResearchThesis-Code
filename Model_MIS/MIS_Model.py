from Model_MIS import MIS_LoadData
#from Model_MIS import RollerCoaster_LossFunction
from Meta_Model import Meta_Model
import random
import re
import numpy as np


class Model_MIS(Meta_Model.MetaModel):
    def __init__(self):
        filename = '../Model_MIS/MIS_PM.xlsx'
        activities, events = MIS_LoadData.loadData(filename)
        defaultLossFunction = lambda tupl: 100-1.*sum(tupl)/len(tupl)
        #defaultLossFunction = lambda tupl: 100-1* tupl[2]



        self.simulate = self.simulateStepwise_withEvents #the normal simulation is here the one with events
        self.baseBudget = int(1.1 * pow(10,6))
        self.baseScoreQuality = 50
        self.startCost = sum(act.variants[0].base_cost for act in activities)



        super().__init__(activities,defaultLossFunction,self.calcPerformanceFunction,events)
        self.targetDuration = 330
        self.resetFunction()

    def calcPerformanceFunction(self,activities=[]):
        if activities == []:
            activities = self.activities
        #totalCost = sum([act.variants[0].cost for act in activities])+self.projectCost
        return (self.ScoreCost(),self.ScoreTime(),self.ScoreQuality(),self.ScoreAcceptance,self.ScoreMorale,self.ScoreSecurity)

    def resetFunction(self):
        self.cost = self.startCost
        self.budget = self.baseBudget
        self.quality = 0
        self.delay = 0

        self.ScoreAcceptance = 50
        self.ScoreMorale = 50
        self.ScoreSecurity = 50

        self.boughtResources = []
        self.projectCost = 0
        self.budget = int(1100)

        for act in self.activities:
            act.resetFunction()
        for event in self.events.values():
            event.resetFunction()

    def criticalPathLength(self):
        for activity in self.activities:
            if activity.finished == True:
                activity.expectedEndpoint = activity.startpoint + activity.variants[0].duration
            if len(activity.predecessors) == 0:
                startpoint = 0
            else:
                startpoint = max(act.expectedEndpoint for act in activity.predecessors)
            activity.expectedEndpoint = startpoint + activity.variants[0].duration
        return activity.expectedEndpoint




    def ScoreCost(self):
        costScale = 30
        scoreCost = 50 + (self.budget - self.cost)*1.0/costScale
        return scoreCost

    def ScoreTime(self):
        timeScale = 5
        actualDuration = self.criticalPathLength()
        scoreTime = 50 + (self.targetDuration - actualDuration)*1.0 /timeScale
        return scoreTime

    def ScoreQuality(self):
        meanActivityQuality = sum(act.variants[0].quality for act in self.activities)/len(self.activities)
        scoreQuality = 50 +meanActivityQuality*1.0*10
        return scoreQuality




    def TaskTime(self,taskIndex):
        variant = self.activities[taskIndex-1].variants[0]
        tasktime = variant.progress
        return tasktime

    def TaskOvertime(self,taskIndex):
        variant = self.activities[taskIndex - 1].variants[0]
        overtime = variant.progress - variant.base_duration
        return overtime

    def OnEventList(self,eventID):
        event = self.events[eventID-1]
        #isOnEventList = all(event.occurCondition(Struct()))
        #return False

    def BoughtResource(self,resourceNumber):
        return resourceNumber in self.boughtResources

    def ResourceAvailable(self,resourceNumber):
        return resourceNumber not in self.boughtResources

    def RandomPercent(self,model):
        return random.randrange(0,100)




    def action_Event(self,eventID):
        event = self.events[eventID]
        event.isActivated = True
        scheduledTime = self.Time + event.TimeLag
        timeCondition = lambda model: model.Time == scheduledTime
        #print("scheduled event " + str(eventID) + " to happen on " + str(scheduledTime))
        event.occurConditions_basic.append(timeCondition)

    def action_ProjectDelay(self,noDays):
        self.TimeDelay += noDays

    def action_TaskQuality(self, taskID,num):
        variant = self.activities[taskID-1].variants[0]
        variant.quality +=num

    def action_ScoreAcceptance(self, num):
        self.ScoreAcceptance+=num

    def action_ScoreSecurity(self, num):
        self.ScoreSecurity+=num

    def action_ScoreMorale(self, num):
        self.ScoreMorale += num

    def action_TaskCost(self,taskID,num):
        variant = self.activities[taskID-1].variants[0]
        variant.cost += num

    def action_ProjectCost(self,num):
        self.projectCost += num

    def action_TaskDelay(self,taskID,noDays):
        variant = self.activities[taskID - 1].variants[0]
        variant.duration += noDays

    def action_CancelEvent(self,eventID):
        event = self.events[eventID]
        event.isActivated = False

    def action_Budget(self,num):
        self.budget += num

    def action_CurrTaskQual(self,num):
        currTask = next(task for task in self.activities if not task.finished)
        currTask.variants[0].quality+=num

    def action_BoughtResource(self,num):
        self.boughtResources.append(num)



class Activity_MIS(Meta_Model.Activity):
    def __init__(self,activityID,variants,name):
        self.name = name
        self.predecessors =[]
        super().__init__([],variants,activityID)

class Variant_MIS(Meta_Model.Variant):
    def __init__(self,base_duration,base_cost):
        self.base_duration = base_duration
        self.duration = base_duration
        self.base_cost = base_cost
        self.cost = base_cost
        self.quality = 0
        self.progress = -1

        super().__init__(self.simulate,self.simulateStepFunction)

    def resetFunction(self):
        self.duration = self.base_duration
        self.cost = self.base_cost
        self.quality = 0
        self.progress = -1
        super().resetFunction()

    def simulate(self):
        return [0,0,0,0,0,0] #for calculating Startpoints


    def simulateStepFunction(self,model):
        self.ensureStartpoint()
        self.progress +=1
        relProgress = self.progress*1.0/self.duration
        if relProgress >=1:
            self.activity.endpoint = self.activity.startpoint+self.duration
        return relProgress

    def ensureStartpoint(self):
        if not hasattr(self.activity,'startpoint'):
            if len(self.activity.predecessors) > 0:
                self.activity.startpoint = max(pred.endpoint for pred in self.activity.predecessors)
            else:
                self.activity.startpoint = int(0)
            self.progress = 0



