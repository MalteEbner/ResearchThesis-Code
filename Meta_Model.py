import numpy as np
'''
general Meta-Model
'''

class Activity():
    def __init__(self, predecessors, variants):
        self.predecessors = predecessors
        self.variants = variants
    def chooseVariant(self, variantIndex):
        self.variantID = variantIndex
        self.chosenVariant = self.variants[variantIndex]


class Variant():
    def __init__(self, simulate, simulateStep=[]):
        self.simulate = simulate
        self.simulateStep = simulateStep



'''
Meta-Model for the refinery
'''

class Activity_refinery(Activity):
    def __init__(self,predecessors,suppliers,type,base_duration,base_cost_per_day,name,activity_ID):

        self.name = name
        self.type = type
        self.base_duration = base_duration
        self.base_cost_per_day = base_cost_per_day
        self.activity_ID = activity_ID

        simulateFunctions = [(lambda y: ( lambda: simulate_refinery(self, y)))(supplier) for supplier in suppliers]
        variants = [Variant(simulateFunction) for simulateFunction in simulateFunctions]
        super().__init__(predecessors,variants)


class Supplier():
    def __init__(self,name,competences):
        self.name = name
        self.competences = competences

    def hasCompetenceType(self,type):
        hasCompType = type in self.competences
        return hasCompType

class Compentence():
    def __init__(self,durationEfficiency,costEfficiency,qualityEfficiency):
        self.durationEfficiency = durationEfficiency
        self.costEfficiency = costEfficiency
        self.qualityEfficiency = qualityEfficiency


def simulate_refinery(activity_refinery,supplier):
    competence = supplier.competences[activity_refinery.type]
    if len(activity_refinery.predecessors)>0:
        predecessorQualities = [pred.quality for pred in activity_refinery.predecessors]
        averagePredecessorQuality = np.mean(predecessorQualities)
    else:
        averagePredecessorQuality = 1
    quality = 0.75 * competence.qualityEfficiency + 0.25 * averagePredecessorQuality
    duration = np.ceil(activity_refinery.base_duration /(competence.durationEfficiency*quality**2))
    cost = np.ceil(activity_refinery.base_cost_per_day* duration/competence.costEfficiency)
    activity_refinery.quality = quality
    return duration,cost,quality


