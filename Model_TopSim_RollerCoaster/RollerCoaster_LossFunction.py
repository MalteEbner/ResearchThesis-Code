import pandas as pd


class RollerCoaster_Loss:
    def __init__(self,filename):
        self.readExcel(filename)


    def readExcel(self,filename):
        sheet = pd.read_excel(filename, sheet_name='Losses')

        DurationLossList = []
        for rowIndex in range(1,130):
            loss = sheet[4][rowIndex]
            DurationLossList.append(loss)

        TechnologyLossList = []
        for rowIndex in range(1,130):
            loss = sheet[8][rowIndex]
            TechnologyLossList.append(loss)

        QualityLossList = []
        for rowIndex in range(1,130):
            loss = sheet[12][rowIndex]
            QualityLossList.append(loss)

        self.DurationLossList = DurationLossList
        self.TechnologyLossList = TechnologyLossList
        self.QualityLossList = QualityLossList

    def RollerCoaster_calcLoss(self, performance):
        duration = performance[0]
        cost = performance[1]
        technology = performance[2]
        quality = performance[3]
        revenue = 8050

        defaultDuration = int(65)
        lossZeroPoint = int(52)
        lossDuration = self.DurationLossList[int(duration-defaultDuration+lossZeroPoint)]
        lossTechnical = self.TechnologyLossList[int(-1*technology+lossZeroPoint)]
        lossQuality = self.QualityLossList[int(-1*quality + lossZeroPoint)]
        totalLoss = cost + lossDuration + lossTechnical + lossQuality

        profit = revenue - totalLoss
        return -1 * profit













