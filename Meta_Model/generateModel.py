from Model_Refinery.Refinery_Model import Model_Refinery
from Model_TopSim_RollerCoaster.RollerCoaster_Model import Model_RollerCoaster
from Model_MIS.MIS_Model import Model_MIS


def generateModel(modelOptions):
    if modelOptions.type == 'RollerCoaster':
        return Model_RollerCoaster(modelOptions)
    if modelOptions.type == 'MIS':
        modelOptions.probabilistic = False
        modelOptions.withEvents = True
        return Model_MIS(modelOptions)
    if modelOptions.type == "Refinery":
        modelOptions.probabilistic = False
        modelOptions.withEvents = False
        return Model_Refinery(modelOptions)