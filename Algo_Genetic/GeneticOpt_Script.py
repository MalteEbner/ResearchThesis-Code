from Algo_Genetic import GeneticOpt_Class

from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model

#model = RollerCoaster_Model.Model_RollerCoaster()
#model = Refinery_Model.Model_Refinery()
model = MIS_Model.Model_MIS()

activityVariantNumbers = model.getVariantNumbers()
genePool = GeneticOpt_Class.GeneticOpt(activityVariantNumbers,model.simulate_returnLoss,5000)
startChromosome = model.getZeroStartpoint()
print(model.simulateMean(startChromosome))
genePool.population.append(startChromosome)
for i in range(100):
    best = genePool.generateNewPop()
    print(str(i) + ":  " + str(model.simulateMean(best)))


best = genePool.getBestPop(model.simulateMean_returnLoss)
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








