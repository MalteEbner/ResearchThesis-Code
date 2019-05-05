from Model_MIS import MIS_Model


model = MIS_Model.Model_MIS()
startpoint = model.getZeroStartpoint()
print('Starting with simulation:')
model.simulateStepwise_withEvents(startpoint)

print("\n\n")
print("performance: " + str(model.calcPerformanceFunction()))