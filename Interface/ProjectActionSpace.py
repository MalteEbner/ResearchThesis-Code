from Interface.ActionSpace import ActionSpace,Action
from gym import spaces

class ProjectActionSpace(ActionSpace):
    def __init__(self,activityVariantNumbers,eventVariantNumbers=[],withScheduleCompression=False):
        # define Action Spaces using the 'spaces' class provided by the openAI gym
        activityVariantSpace = spaces.MultiDiscrete(activityVariantNumbers)
        eventVariantSpace = spaces.MultiDiscrete(eventVariantNumbers)
        spacesTuple = (activityVariantSpace, eventVariantSpace)
        if withScheduleCompression:
            scheduleCompressionSpace = spaces.Box(0.5, 1, shape=(len(activityVariantNumbers),))
            spacesTuple += (scheduleCompressionSpace,)
        super().__init__(spacesTuple)

    def sample(self):
        action = super().sample()
        projectAction = ProjectAction(self)
        projectAction.valuesList = action.valuesList
        return projectAction

class ProjectAction(Action):
    def __init__(self,projectActionSpace):
        if not isinstance(projectActionSpace,ProjectActionSpace):
            print("Expected argument to be instance of ProjectActionSpace, but is %s instead" % type(projectActionSpace))
        super().__init__(projectActionSpace)


    def eventIndizesAsDict(self,orderedEventIDs):
        eventIndizesDict = dict(zip(orderedEventIDs,self.valuesList[1]))
        return eventIndizesDict

    def activityIndizes(self):
        return self.valuesList[0]

    def eventIndizes(self):
        return self.valuesList[1]

    def scheduleCompressionFactors(self):
        if len(self.actionSpace.spaces)==3:
            return self.valuesList[2]
        else:
            raise ValueError



    def __repr__(self):
        lines = []
        spaceNames = ["activities:", "events:", "scheduleCompressionFactors:"]
        for values, index in enumerate(self.valuesList):
            lines += spaceNames[index]
            lines += str(values)
            lines += "\n"
        return lines