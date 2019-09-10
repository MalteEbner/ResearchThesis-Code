import random
import numpy as np
from Model_general import commonFunctions
from Interface import ActionSpace


class GeneticOpt():

    def __init__(self,actionSpace,lossFunction,initialPopSize=500):

        self.variantNumbers = actionSpace.VariantNumbers()
        self.noVariants = len(self.variantNumbers)
        self.actionSpace = actionSpace
        self.lossFunction = lossFunction

        self.initializePopulation(initialPopSize)


        self.defaultMutateProb = 1.0/15



    def initializePopulation(self,popSize):
        self.population = [Chromosome(self.actionSpace.sampleAction()) for i in range(popSize-1)]
        self.appendToPop(self.actionSpace.sampleZeroAction())

    def appendToPop(self,action):
        self.population.append(Chromosome(action))

    def generateNewPop(self):
        losses = [self.lossFunction(chrom) for chrom in self.population]
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
        self.saveDirectly(action.activityIndizes,action.eventIndizes,action.scheduleCompressionFactors)

    def getCopiedChrom(self):
        newAction = ActionSpace.Action(self.actionSpace)
        newAction.saveDirectly(self.activityIndizes.copy(),self.eventIndizes.copy(),self.scheduleCompressionFactors.copy())
        newChrom = Chromosome(newAction)
        return newChrom

    def mutate(self,mutateProb):

        newChrom = self.getCopiedChrom()
        newChrom.mutateWithoutCopying(mutateProb)

        return newChrom

    def mutateWithoutCopying(self,mutateProb):
        activityVariantNumbers = self.actionSpace.ActivityVariantNumbers()
        for i in range(self.actionSpace.noActivities):
            if random.random() < mutateProb:
                self.activityIndizes[i] = random.randrange(activityVariantNumbers[i])
            if self.actionSpace.withScheduleCompression:#random.random() < mutateProb:
                newFactor = self.scheduleCompressionFactors[i] + random.uniform(-0.1,0.1)
                newFactor = max(min(newFactor,1),0.5)
                self.scheduleCompressionFactors[i] = newFactor

        eventVariantNumbers = self.actionSpace.EventVariantNumbers()
        for i in range(self.actionSpace.noEvents):
            if random.random() < mutateProb:
                self.eventIndizes[i] = random.randrange(eventVariantNumbers[i])

        return self


    def cross(self, chromosomeB, probB=0.5):
        newChrom = self.getCopiedChrom()

        for i in range(newChrom.actionSpace.noActivities):
            #mutate activityIndex, scheduleCompression combination together
            if random.random() < probB:
                newChrom.activityIndizes[i] = chromosomeB.activityIndizes[i]
                if self.actionSpace.withScheduleCompression:
                    newChrom.scheduleCompressionFactors[i] = chromosomeB.scheduleCompressionFactors[i]


        for i in range(newChrom.actionSpace.noEvents):
            if random.random() < probB:
                newChrom.eventIndizes[i] = chromosomeB.eventIndizes[i]

        return newChrom







