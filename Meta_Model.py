
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
    def __init__(self,predecessors,suppliers,type,base_duration,base_cost_per_day):
        self.type = type
        self.base_duration = base_duration
        self.base_cost_per_day = base_cost_per_day
        simulateFunctions = [lambda: simulate_refinery(self,supplier) for supplier in suppliers]
        variants = [Variant(simulateFunction) for simulateFunction in simulateFunctions]
        Activity(predecessors,variants)

class Supplier()
    def __init__(self,name,competences):
        self.name = name
        self.competences = competences

class Compentence()
    def __init__(self,type,duration,cost,quality):
        self.type = type
        self.duration = duration
        self.cost = cost
        self.quality = quality


import numpy

def simulate_refinery(activity_refinery,supplier):
    competence = supplier[activity_refinery.type]
    averagePredecessorQuality = numpy.mean(pred.quality for pred in activity_refinery.predecessors)
    quality = 0.75 * competence.quality + 0.25 * averagePredecessorQuality
    duration = activity_refinery.base_duration /(competence.duration*quality^2)
    cost = activity_refinery.base_cost_per_day/competence.cost * duration
    return duration,cost,quality


