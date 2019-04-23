from Model_Refinery import Refinery_Model

'''
load data from excel
'''
import pandas as pd

def loadSupplierData(filename):
    sheet = pd.read_excel(filename, sheet_name='Suppliers')
    #print(sheet)

    competenceTypes = []
    for competenceIndex in range(10):
        columnIndex = round(3 + competenceIndex * 3)
        type = sheet[columnIndex][1].strip()
        competenceTypes.append(type)

    suppliers =[]
    for supplierIndex in range(11):
        rowNumber = 3 + supplierIndex
        supplierName = sheet[2][rowNumber].strip()
        competences = []
        for competenceIndex in range(10):
            columnIndex = 3 + competenceIndex * 3
            durationEfficiency = sheet[columnIndex][rowNumber]
            if durationEfficiency > 0: #only add competence if supplier has it
                costEfficiency = sheet[columnIndex+1][rowNumber]
                qualityEfficiency = sheet[columnIndex+ 2][rowNumber]
                competence = Refinery_Model.Compentence(durationEfficiency, costEfficiency, qualityEfficiency)
                competences.append((competenceTypes[competenceIndex],competence))
        competences = dict(competences)
        supplier = Refinery_Model.Supplier(supplierName, competences)
        suppliers.append(supplier)

    return suppliers


#suppliers = loadSupplierData()
#print(suppliers)