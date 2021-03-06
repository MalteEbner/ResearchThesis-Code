from Model_Refinery import Refinery_Model
'''
load data from excel
'''
import pandas as pd

def loadActivityData(allSuppliers, filename):
    sheet = pd.read_excel(filename, sheet_name='Simulation')

    activities = []
    for activityIndex in range(5,32):
        activityNumber = sheet[1][activityIndex]
        if activityNumber>0:
            name = sheet[2][activityIndex].strip()
            predecessorIndices = sheet[3][activityIndex]
            if type(predecessorIndices)==type('string'): # multiple predecessors
                predecessorIndices = predecessorIndices.split(';')
                predecessorIndices = [int(ind)-1 for ind in predecessorIndices]
            elif predecessorIndices > 0: # one predecessor
                predecessorIndices = [int(predecessorIndices)-1]
            else:
                predecessorIndices = [] # no predecessor


            activityType = sheet[4][activityIndex].strip()
            base_duration = sheet[6][activityIndex]
            base_cost_per_day = sheet[8][activityIndex]
            suppliers = [supplier for supplier in allSuppliers if supplier.hasCompetenceType(activityType)]
            activity = Refinery_Model.Activity_refinery(predecessorIndices, suppliers, activityType, base_duration, base_cost_per_day, name, activityNumber)
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



    return orderedActivities





