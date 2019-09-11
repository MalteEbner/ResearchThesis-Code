


class Model_options():
    def __init__(self, projectType, probabilistic=False, withEvents=False, withScheduleCompression = False, RLcapable=False,
                 interface='Default'):


        self.projectType = projectType
        self.probabilistic = probabilistic
        self.withEvents = withEvents
        self.withScheduleCompression = withScheduleCompression
        self.RLcapable = RLcapable
        self.interface = interface


