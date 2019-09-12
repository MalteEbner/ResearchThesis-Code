import random
import numpy as np
from Model_general import commonFunctions
from Interface import ActionSpace
from gym import spaces


class GeneticOpt():

    def __init__(self,actionSpace,lossFunction,initialPopSize=500):

        self.actionSpace = actionSpace
        self.lossFunction = lossFunction

        self.initializePopulation(initialPopSize)


        self.defaultMutateProb = 1.0/15



    def initializePopulation(self,popSize):
        self.population = [Chromosome(self.actionSpace.sample()) for i in range(popSize-1)]

    def appendToPop(self,action):
        self.population.append(Chromosome(action))

    def generateNewPop(self):
        losses = self.lossFunction(self.population)
        soFarBestChromosome = self.population[np.argmin(losses)]
        matePopIndices, elitePopIndices = self.selection(losses)
        self.population = self.breedPopulation(matePopIndices,elitePopIndices)
        return soFarBestChromosome

    def selection(self,losses,matePoolSize=20,eliteSize=5):
        exploitationFactor = 0.1
        probs = commonFunctions.probsFromLosses(losses,exploitationFactor)
        try:
            matePopIndices = np.random.choice(range(len(probs)), matePoolSize, replace=False, p=probs)
        except ValueError:
            print('ERROR: probs either NaN or non-negative:')
            print('losses: ' + str(losses))
            print('probs: ' + str(probs))
            raise ValueError
        elitePopIndices = np.argpartition(losses, eliteSize)[:eliteSize]
        #matePopIndices = set(matePopIndices)-set(elitePopIndices)
        return matePopIndices,elitePopIndices

    def breedPopulation(self,matePopIndices, elitePopIndices):
        elitePop = [self.population[i] for i in elitePopIndices]
        matePop = [self.population[i] for i in matePopIndices]

        totalParentPop = elitePop[:]
        totalParentPop.extend(matePop)

        '''
        Child Population consists of
        a) elitePop, unchanged
        b) elitePop, mutated
        c) totalParentPop, crossed & mutated
        '''

        #a)
        childPop = elitePop[:]

        #b)
        for elite in elitePop:
            mutated_elite = elite.mutate(self.defaultMutateProb)
            childPop.append(mutated_elite)

        #c)
        while len(childPop) < len(self.population):
            mateAindex = random.randrange(len(totalParentPop))
            mateBindex = random.randrange(len(totalParentPop))
            mateA = totalParentPop[mateAindex]
            mateB = totalParentPop[mateBindex]
            crossed_child = mateA.cross(mateB)
            child = crossed_child.mutateWithoutCopying(self.defaultMutateProb)
            childPop.append(child)

        return childPop

    def getBestPop(self,losses=[],lossFunction=[]):
        if lossFunction==[]:
            lossFunction = self.lossFunction
        if losses==[]:
            losses = [lossFunction(chrom) for chrom in self.population]
        bestIndex = np.argmin(losses)
        return self.population[bestIndex]


class Chromosome(ActionSpace.Action):
    def __init__(self,action):
        super().__init__(action.actionSpace)
        self.valuesList = action.valuesList

    def getCopiedChrom(self):
        newChrom = Chromosome(self)
        copiedValuesList = [values.copy() for values in self.valuesList]
        newChrom.valuesList = copiedValuesList
        return newChrom

    def mutate(self,mutateProb):

        newChrom = self.getCopiedChrom()
        newChrom.mutateWithoutCopying(mutateProb)

        return newChrom

    def mutateWithoutCopying(self,mutateProb):
        valuesList = []
        for space ,values in zip(self.actionSpace.spaces,self.valuesList):
            if isinstance(space, spaces.MultiDiscrete):
                for index,noVariants in enumerate(space.nvec):
                    if random.random() < mutateProb and noVariants>1:
                        values[index] = random.randrange(0,noVariants-1)
            elif isinstance(space, spaces.Box):
                if space.is_bounded():
                    values += np.random.uniform(-0.1,0.1,len(values))
                    values = np.maximum(space.low, np.minimum(space.high, values))
                else:  # assumes space is unbounded on both sides
                    values += np.random.normal(loc=0,scale=0.1,size=len(values))


            else:
                raise NotImplementedError
            valuesList.append(values)
        self.valuesList = valuesList
        return self


    def cross(self, chromosomeB, probB=0.5):
        newChrom = self.getCopiedChrom()
        valuesList = []
        for space, valuesA , valuesB in zip(self.actionSpace.spaces, newChrom.valuesList, chromosomeB.valuesList):
            for i in range(len(valuesA)):
                if random.random() < 0.5:
                    valuesA[i] = valuesB[i]
                    if valuesA[i]>5:
                        i=1
            valuesList.append(valuesA)
        newChrom.valuesList = valuesList
        return newChrom







