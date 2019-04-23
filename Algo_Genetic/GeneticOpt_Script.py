from Model_Refinery import Refinery_Model
from Algo_Genetic import GeneticOpt_Class


model = Refinery_Model.Model_Refinery()
activityVariantNumbers = model.getVariantNumbers()
genePool = GeneticOpt_Class.GeneticOpt(activityVariantNumbers,model.simulate_returnLoss)
startChromosome = model.getStartpoint()
genePool.population.append(startChromosome)
for i in range(100):
    best = genePool.generateNewPop()
    print(model.simulate(best))

'''
best = genePool.getBestPop()
print(model.simulate(best))
'''







