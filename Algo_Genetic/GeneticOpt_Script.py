from Algo_Genetic import GeneticOpt_Class
from Interface.generateModel import generateModel
import time
from Interface.Model_options import Model_options
import random


'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'Refinery' , 'MIS' or 'RollerCoaster'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=True
  modelOptions.interface='VAE'
model = generateModel(modelOptions)


'''define start Population of genetic algorithm'''
actionSpace = model.getActionSpace()
genePool = GeneticOpt_Class.GeneticOpt(actionSpace,model.simulate_returnLoss_onBatch,100)


'''run genetic algorithm'''
start = time.time()
for i in range(3600):
    bestAction = genePool.generateNewPop()
    if i%1 == 0:
        print(str(i) + ":  " + str(model.simulate(bestAction)) + ' time:' + str(time.time()-start))
end = time.time()
print('time needed: ' + str(end-start))


'''print best chromosome'''
best = genePool.getBestPop()
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








