from Model_MIS import MIS_Model
import numpy as np

'''TODO: adapt to new generate-model API'''

model = MIS_Model.Model_MIS()
startpoint = [num-1 for num in model.getActionSpace().VariantNumbers()]
print('noPossiblePoints: '+ str(float(np.prod(model.getVariantNumbers()))))
print('Starting with simulation')
model.simulateStepwise_withEvents(startpoint)

print("noEventsPlayed:" + str(model.noEventsPlayed))
print("performance: " + str(model.calcPerformanceFunction()))
print("\n\n")
print("\n\n")
print("\n\n")
print("\n\n")
i=0