


class Model_options():
    def __init__(self, type, probabilistic=False, withEvents=False, withScheduleCompression = False, RLcapable=False):
        self.type = type
        self.probabilistic = probabilistic
        self.withEvents = withEvents
        self.withScheduleCompression = withScheduleCompression
        self.RLcapable = RLcapable


