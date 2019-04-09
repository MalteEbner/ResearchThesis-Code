class activity():
    def __init__(self, predecessors, variants):
        self.predecessors = predecessors
        self.variants = variants
    def chooseVariant(self, variantID):
        self.variantID = variantID
        self.chosenVariant = self.variants[variantID]


class variant():
    def __init__(self, simulate, simulateStep):
        self.simulate = simulate
        self.simulateStep = simulateStep


