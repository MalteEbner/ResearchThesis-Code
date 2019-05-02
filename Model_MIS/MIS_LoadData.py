from Model_MIS import MIS_Model
'''
load data from excel
'''
import pandas as pd

def loadData(filename):
    sheet = pd.read_excel(filename, sheet_name='Schedule')


    #replace predecessor indices by class instances
    for activity in activities:
        activity.predecessors = [activities[ind] for ind in activity.predecessors]

    #order activities such that each activity is after all precedessors
    orderedActivities = []
    while len(orderedActivities)<len(activities):
        for activity in activities:
            if not activity in orderedActivities and all(act in orderedActivities for act in activity.predecessors):
                orderedActivities.append(activity)



    return orderedActivities

def readTasks(filename):
    sheet = pd.read_excel(filename, sheet_name='tasks')
    activities = []
    for task_ID,task_name,mean_dur,mean_cost,predecessorIndices in zip(sheet['TaskID'],sheet['Task_Name'],sheet['Mean_Task_Duration'],sheet['Mean_Task_Cost'],sheet['Predecessors']):
        if type(predecessorIndices) == type('string'):  # multiple predecessors
            predecessorIndices = predecessorIndices.split(';')
            predecessorIndices = [int(ind) - 1 for ind in predecessorIndices]
        elif predecessorIndices > 0:  # one predecessor
            predecessorIndices = [int(predecessorIndices) - 1]
        else:
            predecessorIndices = []  # no predecessor
        activity = MIS_Model.Activity_MIS(task_ID,task_name,mean_dur,mean_cost)
        activity.predecessors = predecessorIndices
        activities.append(activity)

    #replace predecessor indices by class instances
    for activity in activities:
        activity.predecessors = [activities[ind] for ind in activity.predecessors]

    #order activities such that each activity is after all predecessors
    orderedActivities = []
    while len(orderedActivities)<len(activities):
        for activity in activities:
            if not activity in orderedActivities and all(act in orderedActivities for act in activity.predecessors):
                orderedActivities.append(activity)

    return activity









