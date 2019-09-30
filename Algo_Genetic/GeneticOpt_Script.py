from Algo_Genetic import GeneticOpt_Class
from Interface.generateModel import generateModel
import time
from Interface.Model_options import Model_options
import random


'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'Refinery' , 'MIS' or 'RollerCoaster'
#modelOptions.probabilistic = True
modelOptions.withScheduleCompression=True
#modelOptions.interface='VAE'
model = generateModel(modelOptions)


'''define start Population of genetic algorithm'''
actionSpace = model.getActionSpace()

popSize = 100
genePool = GeneticOpt_Class.GeneticOpt(actionSpace,model.simulate_returnLoss_onBatch,popSize)



'''run genetic algorithm'''
start = time.time()
noIters = 3
for i in range(noIters):
    bestAction = genePool.generateNewPop()
    if i%1 == 0:
        print(str(i) + ":  " + str(model.simulate(bestAction)) + ' time:' + str(time.time()-start))
end = time.time()
print('time needed: ' + str(end-start))


'''print best chromosome'''
best = genePool.getBestPop()
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))

noSamples = noIters * popSize
model.savePerformance('actorCritic',end-start,noSamples,best)

print("Finished")








