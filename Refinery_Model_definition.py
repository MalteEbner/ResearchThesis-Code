import Refinery_Model_ProjectNetwork
import Refinery_Model_suppliers

suppliers = Refinery_Model_suppliers.loadSupplierData()
activities = Refinery_Model_ProjectNetwork.loadActivityData(suppliers)


for activity in activities:
    blub = activity.variants[0].simulate()
    print(blub)



