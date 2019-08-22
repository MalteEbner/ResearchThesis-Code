'''Model Imports'''
from Interface.generateModel import generateModel
from Interface.Model_options import Model_options
from Interface import ActionSpace

'''Optimization imports'''
### Necessary imports
import time
import GPyOpt


'''generate Model with its options'''
modelOptions = Model_options('Refinery') #type: 'RollerCoaster' , 'MIS' or 'Refinery'
modelOptions.probabilistic = False
modelOptions.withScheduleCompression=False
model = generateModel(modelOptions)


'''define parameter space'''
actionSpace = model.getActionSpace()
action = ActionSpace.Action(actionSpace)
mixed_domain = []
for index,varNumber in enumerate(actionSpace.VariantNumbers()):
    options = range(varNumber)
    variable = {'name':'var_'+str(index),
                'type':'categorical',
               'domain': range(varNumber)}
    mixed_domain.append(variable)

if modelOptions.withScheduleCompression:
    for index in range(actionSpace.noActivities):
        variable = {'name':'compression_var_'+str(index),
                    'type':'continuous',
                    'domain':(0.5,1)}
        mixed_domain.append(variable)


'''define objective function '''
def objective_function(input):
    action.saveEverythingCombined(input[0])
    loss = model.simulate_returnLoss(action)
    return loss

myBopt = GPyOpt.methods.BayesianOptimization(f=objective_function,                     # Objective function
                                             domain=mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = 5,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True)           # True evaluations, no sample noise
print('starting optimization')
start = time.time()
myBopt.run_optimization(max_iter=600,eps=-1,verbosity=True)
end = time.time()
print('time needed: ' + str(end-start))


myBopt.plot_convergence()

bestX = myBopt.x_opt
action.saveEverythingCombined(bestX)
print("best input: " + str(bestX))
print("performance of best: " + str(model.simulateMean(action)))
print("end of optimization")

