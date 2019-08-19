from Algo_ActorCritic import ActorCritic_Class
from Meta_Model.generateModel import generateModel
import time
from Meta_Model.Meta_Model_options import Meta_Model_options
import numpy as np
from math import log

def actorCritic_RunAlgo(model, verbose = 1, hyperparams=0):
    if verbose >=1:
        print("NEW: run of actor critic algo")

    '''generate Model with its options'''
    modelOptions = Meta_Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
    modelOptions.probabilistic = False
    modelOptions.withScheduleCompression=True
    model = generateModel(modelOptions)


    '''define policy'''
    actionSpace = model.getActionSpace()
    policy = ActorCritic_Class.Policy(actionSpace)

    '''define baseline as average of 100 random actions'''
    losses = [model.simulate_returnLoss(actionSpace.getRandomAction()) for i in range(100)]
    baseline = np.mean(losses)
    baseStd = np.std(losses)
    if verbose >= 1:
        print('baseline:' + str(baseline) + " baseline_std: " + str(baseStd))

    '''hyperparams'''
    batchSize = hyperparams.get('batchSize',8)
    noSamples = hyperparams.get('noSamples',8192)
    noIterations = int(noSamples/batchSize)
    verboseNoIterations = int(128/batchSize)
    baselineUpdateFactor = hyperparams.get('baselineUpdateFactor',0.1)

    explorationFactor = hyperparams.get('explorationFactor',0.1)
    explorationDecayFactor = hyperparams.get('explorationDecayFactor',0.98)

    learningRate = hyperparams.get('learningRate',1)
    learningRateDecayFactor = hyperparams.get('learningRateDecayFactor',0.98)

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

        #update rates
        explorationFactor *= explorationDecayFactor
        learningRate *= learningRateDecayFactor

        #print best loss
        if  verbose >= 2 and (i % verboseNoIterations == 1 or i == noIterations - 1):


            bestAction = policy.getBestAction()
            loss = model.simulate_returnLoss(bestAction)
            print(str(i) + ":  loss:" + str(loss) + '  baseline:' + str(baseline) + ' time: ' + str(time.time()-start))
        if verbose >= 2 and ( i% (verboseNoIterations*5) == 1 or i == noIterations - 1):
            prediction = policy.model.predict(np.ones((1, 1, 1)))
            print('prediction:' + str(prediction))

    end = time.time()


    if verbose >=1:
        print('time needed: ' + str(end-start))

        '''print best action'''
        best = policy.getBestAction()
        print("best chromosome: " + str(best))
        performanceOfBest = str(model.simulateMean(best))
        print("performance of best: " + performanceOfBest)
        print("Finished")
    
    return performanceOfBest[0]








