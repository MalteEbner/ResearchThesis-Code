from Algo_ActorCritic import ActorCritic_Algo
from Meta_Model.generateModel import generateModel
import time
from Meta_Model.Meta_Model_options import Meta_Model_options
import numpy as np


'''generate Model with its options'''
modelOptions = Meta_Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = True
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)


'''define policy'''
actionSpace = model.getActionSpace()
policy = ActorCritic_Algo.Policy(actionSpace)

'''define baseline as average of 10 random actions'''
losses = []
for i in range(10):
    randomAction = actionSpace.getRandomAction()
    loss = model.simulate_returnLoss(randomAction)
    losses.append(loss)
baseline = np.mean(losses)
baseVariance = np.std(losses)
baselineUpdateFactor = 0.1
print('baseline:' + str(baseline))

'''hyperparams'''
learningRate = -1
batchSize = 128

'''run actor-critic algorithm'''
start = time.time()
for i in range(1000):
    #sample actions
    nextAction = policy.getNextAction()
    #appy action on model, sample 'reward' (loss)
    loss = model.simulate_returnLoss(nextAction)
    #update policy
    advantage = (baseline-loss)/baseVariance
    policy.updateModel(nextAction,advantage)
    #update baseline
    baseline += baselineUpdateFactor*(loss-baseline)

    #print best loss
    if i % 100 == 0:
        bestAction = policy.getBestAction()
        loss = model.simulate_returnLoss(bestAction)
        print(str(i) + ":  " + str(loss))

end = time.time()
print('time needed: ' + str(end-start))


'''print best action'''
best = policy.getBestAction()
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








