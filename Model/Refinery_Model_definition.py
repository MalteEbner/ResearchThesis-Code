from Model import Refinery_Model_suppliers, Refinery_Model_ProjectNetwork
from Meta_Model import MetaModel
import numpy as np


class Model_Refinery(MetaModel):
    def __init__(self):
        self.suppliers = Refinery_Model_suppliers.loadSupplierData()
        activities = Refinery_Model_ProjectNetwork.loadActivityData(self.suppliers)
        defaultLossFunction = lambda tupl: tupl[0] * 6000 + tupl[1]
        super().__init__(activities,defaultLossFunction)


'''
model = Model_Refinery()
print("\nModel 1:")
print(model.simulate())
print(model)
print("\nModel 2:")
startpoint = model.getStartpoint()
print(model.simulate(startpoint))
print(model)
'''




