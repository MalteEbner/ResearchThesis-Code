from Interface import ActionSpace, ProjectActionSpace
import numpy as np

class DefaultInterface():
    def __init__(self,projectModel):
        self.projectModel = projectModel
        self.actionSpace = ActionSpace.ActionSpace(projectModel.getActionSpace().spaces)

    def getActionSpace(self):
        return self.actionSpace

    def actionToProjectAction(self,action):
        projectAction = ProjectActionSpace.ProjectAction(self.projectModel.getActionSpace())
        projectAction.valuesList = action.valuesList
        return projectAction

    def simulate(self, action):
        projectAction = self.actionToProjectAction(action)
        performances = self.projectModel.simulate(projectAction)
        return performances


    def simulate_returnLoss(self,*args):
        loss = self.simulate(*args)[0]
        return loss

    def simulate_returnLoss_onBatch(self,actions):
        return [self.simulate(action)[0] for action in actions]

    def simulateMean(self,action):
        performancesList = [self.simulate(action) for i in range(10)]
        meanPerformance = np.mean(performancesList,axis=0)
        meanPerformance = np.round(meanPerformance,2)
        return meanPerformance

    def printAllAboutAction(self,action):
        print('action: ' + str(action))
        print('performances: ' + str(self.simulateMean(action)))

    def savePerformance(self,algoName, duration,noSamples,bestAction):
        import csv

        name = 'perf_automatic.csv'
        from pathlib import Path
        name = Path('../Interface/') / name

        with open(name, mode='a') as wfile:
            file_writer = csv.writer(wfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            perf = [str(i) for i in self.simulateMean(bestAction)]

            line = [str(self.projectModel.modelOptions)]
            line += [str(algoName)]
            line += perf
            line += [str(duration),str(noSamples)]

            file_writer.writerow(line)
            print('Wrote successfully')



