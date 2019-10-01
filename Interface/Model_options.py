


class Model_options():
    def __init__(self, projectType, probabilistic=False, withEvents=False, withScheduleCompression = False,
                 interface='Default'):


        self.projectType = projectType
        self.probabilistic = probabilistic
        self.withEvents = withEvents
        self.withScheduleCompression = withScheduleCompression
        self.interface = interface

    def __repr__(self):
        name = ''
        name += str(self.projectType)
        if self.withEvents:
            name += '_withEvents'
        if self.withScheduleCompression:
            name += '_withScheduleCompression'
        name += '_'+str(self.interface)
        return name

    def asPretrainedVAE_Filename(self,latentDim):
        name = 'pretrainedVAE_'
        name += str(self.projectType)
        if self.withEvents:
            name += '_withEvents'
        if self.probabilistic:
            name += '_probabilistic'
        if self.withScheduleCompression:
            name += '_withScheduleCompression075'
        name+= '_'+str(latentDim)
        name += '.h5'
        from pathlib import Path
        name = Path('../Interface_VAE/') / name
        return name


