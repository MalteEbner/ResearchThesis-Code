from Algo_ActorCritic import ActorCritic_Class
from Interface.generateModel import generateModel
import time
from Interface.Model_options import Model_options
import numpy as np


def actorCritic_RunAlgo(model=0, verbose = 2, hyperparams=0):
    if verbose >=1:
        print("NEW: run of actor critic algo")
    if hyperparams==0:
        hyperparams = {}

    if model == 0:
        '''generate Model with its options'''
        modelOptions = Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
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
    batchSize = hyperparams.get('batchSize',1)
    noSamples = hyperparams.get('noSamples',80000)
    noIterations = int(noSamples/batchSize)
    verboseNoIterations = max(int(1/batchSize),1)
    baselineUpdateFactor = hyperparams.get('baselineUpdateFactor',0.1)

    explorationFactor = hyperparams.get('explorationFactor',0.9)
    explorationDecayFactor = hyperparams.get('explorationDecayFactor',0.99)

    learningRate = hyperparams.get('learningRate',1)
    learningRateDecayFactor = hyperparams.get('learningRateDecayFactor',0.98)

    '''run actor-critic algorithm'''
    if verbose >=1:
        print('starting loop itself')
    start = time.time()
    for i in range(noIterations):
        #sample actions
        actions = [policy.sampleAction() for i in range(batchSize)]
        #appy action on model, sample 'reward' (loss)
        losses = [model.simulate_returnLoss(action) for action in actions]
        #update baseline
        baseline += baselineUpdateFactor*(np.mean(losses)-baseline)
        #update policy
        advantages = (baseline-losses)/baseStd
        advantages -= explorationFactor #keep exploring
        policy.update(actions,advantages)

        #update rates
        explorationFactor *= explorationDecayFactor
        learningRate *= learningRateDecayFactor

        #print best loss
        if  verbose >= 2 and (i % verboseNoIterations == 1 or i == noIterations - 1):


            bestAction = policy.getBestAction()
            loss = model.simulate_returnLoss(bestAction)
            print(str(i) + ":  loss:" + str(loss) + '  baseline:' + str(baseline) + ' time: ' + str(time.time()-start))
        if False and verbose >= 2 and ( i% (verboseNoIterations*5) == 1 or i == noIterations - 1):
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








