from Model_MIS import MIS_Model
from Model_MIS import MIS_Model_Event
'''
load data from excel
'''
import pandas as pd

def loadData(filename):


    activities = readTasks(filename)
    events = readEvents(filename)
    return activities,events


def readEvents(filename):

    # read events with conditions for popup
    sheetEvents = pd.read_excel(filename, sheet_name='events')
    events = {}
    for event_ID, condition, description in zip(sheetEvents['EventID'], sheetEvents['Condition'],sheetEvents['Description']):
        event = MIS_Model_Event.Event_MIS(event_ID,condition,description)
        events[event_ID] = event


    #read options/variants for each event
    sheetOptions = pd.read_excel(filename, sheet_name='options')
    eventKeys = events.keys()
    for event_ID, option_ID, description, ifCond, ThenDo, elseDo,comment in zip(sheetOptions['EventID'], sheetOptions['OptionID'], sheetOptions['Description'], sheetOptions['If'], sheetOptions['Then'], sheetOptions['Else'], sheetOptions['Comment']):
        if event_ID in eventKeys:
            eventOption = MIS_Model_Event.EventOption_MIS(ifCond,ThenDo,elseDo,description,comment)
            events[event_ID].eventOptions.append(eventOption)

    return events




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
        variant = MIS_Model.Variant_MIS(mean_dur,mean_cost)
        activity = MIS_Model.Activity_MIS(task_ID,[variant],task_name)
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

    return activities









