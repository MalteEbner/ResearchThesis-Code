from Algo_Genetic import GeneticOpt_Class
import time

from Model_Refinery import Refinery_Model
from Model_TopSim_RollerCoaster import RollerCoaster_Model
from Model_MIS import MIS_Model

#model = RollerCoaster_Model.Model_RollerCoaster(False)
#model = Refinery_Model.Model_Refinery()
model = MIS_Model.Model_MIS()
print(model.calcPerformanceFunction())

activityVariantNumbers = model.getVariantNumbers()
genePool = GeneticOpt_Class.GeneticOpt(activityVariantNumbers,model.simulate_returnLoss,100)
if hasattr(model,'getGoodStartpoint'):
    startChromosome = model.getGoodStartpoint()
else:
    startChromosome = model.getZeroStartpoint()
print(model.simulateMean(startChromosome))
genePool.population.append(startChromosome)
start = time.time()
for i in range(100):
    best = genePool.generateNewPop()
    print(str(i) + ":  " + str(model.simulateMean(best)))
end = time.time()
print('time needed: ' + str(end-start))


best = genePool.getBestPop(model.simulateMean_returnLoss)
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








