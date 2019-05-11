from Model_TopSim_RollerCoaster import RollerCoaster_Model
'''
load data from excel
'''
import pandas as pd

def loadActivityData(filename,probabilisticModel):
    sheet = pd.read_excel(filename, sheet_name='Schedule')

    activities = []
    followerIndicesMatrix = []
    for activityIndex in range(2,48):
        activityNumber = sheet[1][activityIndex]
        if activityNumber>0:
            name = sheet[2][activityIndex].strip()
            followerIndices = sheet[3][activityIndex]
            if type(followerIndices)==type('string'): # multiple followers
                followerIndices = followerIndices.split(';')
                followerIndices = [int(ind)-1 for ind in followerIndices]
            elif followerIndices > 0: # one follower
                followerIndices = [int(followerIndices)-1]
            else:
                followerIndices = [] # no follower
            followerIndicesMatrix.append(followerIndices)

            variantDataList = []
            for variantIndex in range(4):
                durationColumn = variantIndex*5+4
                duration = sheet[durationColumn][activityIndex]
                if duration > 0: #if there exists a variant
                    cost = sheet[durationColumn+1][activityIndex]
                    technology = sheet[durationColumn+2][activityIndex]
                    quality = sheet[durationColumn+3][activityIndex]
                    risk = sheet[durationColumn+4][activityIndex]
                    variantData = RollerCoaster_Model.VariantData_RollerCoaster(duration,cost,technology,quality,risk)
                    variantDataList.append(variantData)
            activity = RollerCoaster_Model.Activity_RollerCoaster(variantDataList,name,activityNumber,probabilisticModel)
            activities.append(activity)

    #write down predecessors
    for activity,followerIndices, in zip(activities,followerIndicesMatrix):
        for followerIndex in followerIndices:
            activities[followerIndex].predecessors.append(activity)

    #order activities such that each activity is after all precedessors
    orderedActivities = []
    while len(orderedActivities)<len(activities):
        for activity in activities:
            if not activity in orderedActivities and all(act in orderedActivities for act in activity.predecessors):
                orderedActivities.append(activity)



    return orderedActivities





