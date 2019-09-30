'''Model Imports'''
from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
from Interface import ActionSpace
from gym import spaces
import numpy as np

'''Optimization imports'''
### Necessary imports
import time
import GPyOpt



'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
#modelOptions.probabilistic = True
modelOptions.withScheduleCompression=True
#modelOptions.interface = "VAE"
model = generateModel(modelOptions)


'''define parameter space'''
actionSpace = model.getActionSpace()
action = ActionSpace.Action(actionSpace)
mixed_domain = []
for space in actionSpace.spaces:
    if isinstance(space, spaces.MultiDiscrete):
        for index, noVariants in enumerate(space.nvec):
            variable = {'name': 'var_' + str(index),
                        'type': 'categorical',
                        'domain': range(noVariants)}
            mixed_domain.append(variable)
    elif isinstance(space, spaces.Box):
        if space.is_bounded():
            for index,low,high in zip(range(space.shape[0]),space.low,space.high):
                variable = {'name': 'compression_var_' + str(index),
                            'type': 'continuous',
                            'domain': (low, high)}
                mixed_domain.append(variable)
        else:
            for index in range(space.shape[0]):
                variable = {'name': 'compression_var_' + str(index),
                            'type': 'continuous',
                            'domain': (-5, 5)}#bounds needed
                mixed_domain.append(variable)
    else:
        raise NotImplementedError



'''define objective function '''
def objective_function(input):
    action.saveEverythingCombined(input[0])
    loss = model.simulate_returnLoss(action)
    print('loss: ' + str(loss))
    return loss

initialNoSamples = 500
myBopt = GPyOpt.methods.BayesianOptimization(f=objective_function,                     # Objective function
                                             domain=mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = initialNoSamples,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True,
                                             verbosity=True)           # True evaluations, no sample noise
print('starting optimization')
start = time.time()
noIters=200
myBopt.run_optimization(max_iter=noIters,eps=-1,verbosity=True)
end = time.time()
print('time needed: ' + str(end-start))


myBopt.plot_convergence()

bestX = myBopt.x_opt

action.saveEverythingCombined(bestX)
model.printAllAboutAction(action)
model.savePerformance('GPyOpt',end-start,initialNoSamples+noIters,action)



