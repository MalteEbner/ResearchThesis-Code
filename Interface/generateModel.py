from Model_Refinery.Refinery_Model import Model_Refinery
from Model_TopSim_RollerCoaster.RollerCoaster_Model import Model_RollerCoaster
from Model_MIS.MIS_Model import Model_MIS
from Interface import DefaultInterface
from Interface_VAE import VAE_interface


def generateModel(modelOptions):
    if modelOptions.projectType == 'RollerCoaster':
        projectModel = Model_RollerCoaster(modelOptions)
    elif modelOptions.projectType == 'MIS':
        modelOptions.probabilistic = False
        modelOptions.withEvents = True
        projectModel = Model_MIS(modelOptions)
    elif modelOptions.projectType == "Refinery":
        modelOptions.probabilistic = False
        modelOptions.withEvents = False
        projectModel = Model_Refinery(modelOptions)
    else:
        raise ValueError

    if modelOptions.interface == "Default":
        model = DefaultInterface.DefaultInterface(projectModel)
    elif modelOptions.interface == "VAE":
        model = VAE_interface.VAE_Interface(projectModel)
    else:
        raise ValueError

    return model
