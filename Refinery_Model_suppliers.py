import Meta_Model


'''
load data from excel
'''
import pandas as pd

def loadSupplierData(filename='/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/Master ETIT/10. Semester/Forschungsarbeit/Project management paper/SimGame translated - Andre Heleno.xlsx'):
    sheet = pd.read_excel(filename, sheet_name='Suppliers')
    #print(sheet)

    competenceTypes = []
    for competenceIndex in range(10):
        columnLetter = round(4 + competenceIndex * 3)
        type = sheet[int(columnLetter)][int(0)]
        competenceTypes.append(type)

    suppliers =[]
    for supplierIndex in range(11):
        rowNumber = 2 + supplierIndex
        supplierName = sheet[3][rowNumber]
        competences = []
        for competenceIndex in range(10):
            columnLetter = 4 + competenceIndex * 3
            duration = sheet[columnLetter][rowNumber]
            if duration > 0: #only add competence if supplier has it
                cost = sheet[columnLetter+1][rowNumber]
                quality = sheet[columnLetter + 2][rowNumber]
                competence = Meta_Model.Compentence(competenceTypes[competenceIndex],duration,cost,quality)
                competences.append(competence)
        supplier = Meta_Model.Supplier(supplierName,competences)
        suppliers.append(supplier)

    return suppliers


suppliers = loadSupplierData()
print(suppliers)