from Algo_ActorCritic import ActorCritic_Class
from Meta_Model.generateModel import generateModel
import time
from Meta_Model.Meta_Model_options import Meta_Model_options
import numpy as np
from math import log


'''generate Model with its options'''
modelOptions = Meta_Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=True
model = generateModel(modelOptions)


'''define policy'''
actionSpace = model.getActionSpace()
policy = ActorCritic_Class.Policy(actionSpace)

'''define baseline as average of 10 random actions'''
losses = []
for i in range(100):
    randomAction = actionSpace.getRandomAction()
    loss = model.simulate_returnLoss(randomAction)
    losses.append(loss)
baseline = np.mean(losses)
baseStd = np.std(losses)
baselineUpdateFactor = 0.1
print('baseline:' + str(baseline) + " baseline_std: " + str(baseStd))

'''hyperparams'''
batchSize = 8
noIterations = int(32768/batchSize)
verboseNoIterations = int(128/batchSize)
explorationFactor = 0.1
endExplorationFactor  = 0.001
explorationDecayFactor = (endExplorationFactor/explorationFactor)**(1/noIterations)
print('explorationDecayFactor: ' + str(explorationDecayFactor))

'''run actor-critic algorithm'''
start = time.time()
for i in range(noIterations):
    #sample actions
    actions = [policy.getNextAction() for i in range(batchSize)]
    #appy action on model, sample 'reward' (loss)
    losses = [model.simulate_returnLoss(action) for action in actions]
    #update policy
    advantages = (baseline-losses)/baseStd
    advantages -= explorationFactor #keep exploring
    policy.updateModel(actions,advantages)
    #update baseline
    baseline += baselineUpdateFactor*(np.mean(losses)-baseline)

    #print best loss
    if i % verboseNoIterations == 1:
        explorationFactor*=explorationDecayFactor

        bestAction = policy.getBestAction()
        loss = model.simulate_returnLoss(bestAction)
        print(str(i) + ":  loss:" + str(loss) + '  baseline:' + str(baseline) + ' time: ' + str(time.time()-start))
    if i% (verboseNoIterations*5) == -1:
        prediction = policy.model.predict(np.ones((1, 1, 1)))
        print('prediction:' + str(prediction))

end = time.time()
print('time needed: ' + str(end-start))


'''print best action'''
best = policy.getBestAction()
print("best chromosome: " + str(best))
print("performance of best: " + str(model.simulateMean(best)))
print("Finished")








