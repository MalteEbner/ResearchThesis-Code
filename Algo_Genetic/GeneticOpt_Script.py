from Model_Refinery import Refinery_Model
from Algo_Genetic import GeneticOpt_Class
from Model_TopSim_RollerCoaster import RollerCoaster_Model

model = RollerCoaster_Model.Model_RollerCoaster()
#model = Refinery_Model.Model_Refinery()

activityVariantNumbers = model.getVariantNumbers()
genePool = GeneticOpt_Class.GeneticOpt(activityVariantNumbers,model.simulate_returnLoss)
startChromosome = model.getZeroStartpoint()
print(model.simulateMean(startChromosome))
genePool.population.append(startChromosome)
for i in range(100):
    best = genePool.generateNewPop()
    print(model.simulateMean(best))


best = genePool.getBestPop()
print(best)
print(model.simulateMean(best))
print("Finished")








