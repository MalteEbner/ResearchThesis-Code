import re
from Model_MIS import MIS_Model
from Meta_Model import Meta_Model

class MIS_conditionStringsToLambdas():
    @staticmethod
    def conditionString_to_conditionFunction(conditionString,optionalEvent=[]):
        if type(conditionString) != type('string'):
            return []
        simpleCondition = r'(\w+[>=<]\d+)'
        bracketCondtion = r'(\w+\(\d+\)[>=<]\d+)'
        allRegex = [simpleCondition, bracketCondtion]
        allRegex = '|'.join(allRegex)
        foundStrings = re.findall(allRegex,conditionString)
        simpleConditions = [MIS_conditionStringsToLambdas.getConditionFun(string[0],optionalEvent) for string in foundStrings if len(string[0])>0]
        bracketConditions = [MIS_conditionStringsToLambdas.getConditionFun_bracket(string[1],optionalEvent) for string in foundStrings if len(string[1])>0]
        occurConditions_basic = simpleConditions + bracketConditions
        occurConditions_basic = [cond for cond in occurConditions_basic if type(cond) == type(lambda model: 'example')]
        return occurConditions_basic

    @staticmethod
    def getConditionFun(simpleConditionString,optionalEventObject=[]):
        regex = r'(\w+)([>=<])(\d+)'
        conditionPartStrings = re.findall(regex,simpleConditionString)[0]
        attributeString = conditionPartStrings[0]
        comparisonString = conditionPartStrings[1]
        valueString = conditionPartStrings[2]
        value = int(valueString)

        if attributeString == 'TimeLag':
            optionalEventObject.TimeLag = value
        elif attributeString == 'Random':
            cond = lambda model: model.RandomPercent(model) < value #random is always with '<'
            return cond
        else:
            if attributeString == 'Time' or attributeString == 'TimeLag':
                optionalEventObject.isActivated = True
            dict = {}
            if attributeString == 'ScoreCost' or attributeString == 'ScoreTime' or attributeString == 'ScoreQuality':
                dict['>'] = lambda model: getattr(model, attributeString)() > value
                dict['='] = lambda model: getattr(model, attributeString)() == value
                dict['<'] = lambda model: getattr(model, attributeString)() < value
            else:
                dict['>'] = lambda model: getattr(model,attributeString) > value
                dict['='] = lambda model: getattr(model,attributeString) == value
                dict['<'] = lambda model: getattr(model,attributeString) < value
            cond = dict[comparisonString]
            return cond


    @staticmethod
    def getConditionFun_bracket(bracketConditionString,optionalEventObject=[]):
        regex = r'(\w+)\((\d+)\)([>=<])(\d+)'
        conditionPartStrings = re.findall(regex,bracketConditionString)[0]
        attributeString = conditionPartStrings[0]
        indexString = conditionPartStrings[1]
        comparisonString = conditionPartStrings[2]
        valueString = conditionPartStrings[3]
        value = int(valueString)
        index = int(indexString)
        if attributeString == 'TaskState':
            pass #ignore TaskState condition / treat it as non-existent
        else:
            if attributeString == 'TaskTime':
                optionalEventObject.isActivated = True
            dict = {}
            dict['>'] = lambda model: getattr(model,attributeString)(index) > value
            dict['='] = lambda model: getattr(model,attributeString)(index) == value
            dict['<'] = lambda model: getattr(model,attributeString)(index) < value
            return dict[comparisonString]


class Event_MIS(Meta_Model.Event):
    def __init__(self,eventID,conditionString,description):
        self.eventID = eventID
        self.description = description
        self.isActivated = False
        self.TimeLag = 0
        self.conditionString = conditionString

        self.occurConditions_basic = MIS_conditionStringsToLambdas.conditionString_to_conditionFunction(conditionString,self)

        super().__init__(self.occurCondition,eventOptions=[])

    def occurCondition(self,model):
        if len(self.occurConditions_basic) == 0 or not self.isActivated:
            return False
        else:
            if type(self.occurConditions_basic[0]) != type(lambda model: True):
                print(type(self.occurConditions_basic[0]))
                print(type(lambda model: True))
            canOccur = all(condition(model) for condition in self.occurConditions_basic)
            return canOccur

    def resetFunction(self):
        self.isActivated = False
        self.TimeLag = 0
        self.occurConditions_basic = MIS_conditionStringsToLambdas.conditionString_to_conditionFunction(self.conditionString,self)
        super().resetFunction()




class EventOption_MIS(Meta_Model.EventOption):
    def __init__(self,ifCondString,thenDoString,elseDoString,description,comment):
        self.description = description
        self.comment = comment
        if type(ifCondString) != type('string'):
            self.ifConditions = [lambda model: True]
        else:
            self.ifConditions = MIS_conditionStringsToLambdas.conditionString_to_conditionFunction([],ifCondString)
        self.thenActions = self.actionString_to_actionFunction(thenDoString)
        self.elseActions = self.actionString_to_actionFunction(elseDoString)

        super().__init__(self.runFunction)

    def actionString_to_actionFunction(self,actionString):
        if type(actionString) != type('string'):
            return []
        oneArgActionRegex = r'(\w+\(\d+\))'
        twoArgActionRegex = r'(\w+\(\d+\,\d+\))'
        allRegex = [oneArgActionRegex, twoArgActionRegex]
        allRegex = '|'.join(allRegex)
        foundStrings = re.findall(allRegex,actionString)
        oneArgActions = [self.getActionFun_oneArg(string[0]) for string in foundStrings if len(string[0])>0]
        twoArgActions = [self.getActionFun_twoArg(string[1]) for string in foundStrings if len(string[1])>0]
        actions = oneArgActions + twoArgActions
        return actions

    def runFunction(self,model):
        if all(condition(model) for condition in self.ifConditions):
            for action in self.thenActions:
                action(model)
        else:
            for action in self.elseActions:
                action(model)


    def getActionFun_oneArg(self,oneArgActionString):
        regex = r'(\w+)\((\d+)\)'
        actionPartString = re.findall(regex,oneArgActionString)[0]
        actionName = actionPartString[0]
        arg1 = int(actionPartString[1])
        actionFunctionName = 'action_'+actionName
        actionFunction = lambda model: getattr(model,actionFunctionName)(arg1)
        return actionFunction

    def getActionFun_twoArg(self,twoArgActionString):
        regex = r'(\w+)\((\d+)\,(\d+)\)'
        actionPartString = re.findall(regex,twoArgActionString)[0]
        actionName = actionPartString[0]
        arg1 = int(actionPartString[1])
        arg2 = int(actionPartString[2])
        actionFunctionName = 'action_'+actionName
        actionFunction = lambda model: getattr(model,actionFunctionName)(arg1,arg2)
        return actionFunction

    def resetFunction(self):
        super().resetFunction()

