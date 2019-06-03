from Algo_Genetic import GeneticOpt_Class
from Meta_Model.generateModel import generateModel
import time
from Meta_Model.Meta_Model_options import Meta_Model_options


'''generate Model with its options'''
modelOptions = Meta_Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
model = generateModel(modelOptions)


'''define start Population of genetic algorithm'''
actionSpace = model.getActionSpace()
genePool = GeneticOpt_Class.GeneticOpt(actionSpace,model.simulate_returnLoss,500)
if hasattr(model,'getGoodStartpoint'):
    startChromosome = model.getGoodStartpoint(actionSpace)
else:
    startChromosome = actionSpace.getZeroAction()
genePool.appendToPop(startChromosome)
print(model.simulateMean(startChromosome))

'''run genetic algorithm'''
start = time.time()
for i in range(100):
    bestAction = genePool.generateNewPop()
    print(str(i) + ":  " + str(model.simulateMean(bestAction)))
end = time.time()
print('time needed: ' + str(end-start))


'''print best chromosome'''
best = genePool.getBestPop()
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








