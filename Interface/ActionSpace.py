import random
from keras.utils import to_categorical
from keras.backend import expand_dims
from gym import spaces

class ActionSpace(spaces.Tuple):
    def __init__(self,spacesTuple):
        super().__init__(spacesTuple)

    def sample(self):
        sample = super().sample()
        action = Action(self)
        action.saveValuesList(sample)
        return action


class Action:
    def __init__(self,actionSpace):
        self.actionSpace = actionSpace

    def saveValuesList(self,samples):
        if not len(samples) == len(self.actionSpace.spaces):
            print('Input needs one tuple element per Subpace of the action space')
            print('expected %d inputs, but instead %d inputs were given.' % (len(self.actionSpace.spaces),len(samples)))
        else:
            self.valuesList = samples

    def saveEverythingCombined(self,valuesAsOneList):
        self.valuesList = []
        noVarsSoFar = 0
        for space in self.actionSpace.spaces:
            noVars = space.shape[0]
            values = valuesAsOneList[noVarsSoFar:noVarsSoFar+noVars]
            noVarsSoFar += noVars
            if isinstance(space,spaces.MultiDiscrete):
                values = [int(value) for value in values]
            self.valuesList.append(values)



    def encodeOneHot(self):
        action = []
        for space ,values in zip(self.actionSpace.spaces,self.valuesList):
            if isinstance(space, spaces.MultiDiscrete):
                for noVariants, index in zip(space.nvec,values):
                    encoding = to_categorical(index, num_classes=noVariants)
                    encoding = expand_dims(encoding, axis=0)
                    encoding = expand_dims(encoding, axis=0)
                    action.append(encoding)
            elif isinstance(space, spaces.Box):
                action.append(values)
            else:
                raise NotImplementedError
        return action

    def checkIfValuesInRange(self):
        for space, values in zip(self.actionSpace.spaces, self.valuesList):
            if isinstance(space, spaces.MultiDiscrete):
                for noVariants, value in zip(space.nvec, values):
                    if value >= noVariants:
                        return False
            elif isinstance(space, spaces.Box):
                if space.is_bounded():
                    if any(values < space.low) or any(values > space.high):
                        return False
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        return True



    def __repr__(self):
        lines = "values:"
        lines += str(self.valuesList)
        return lines


