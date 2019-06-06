import random
import numpy as np
from Meta_Model import commonFunctions
from Meta_Model import ActionSpace

class GeneticOpt():

    def __init__(self,actionSpace,lossFunction,initialPopSize=200):

        self.variantNumbers = actionSpace.activityVariantNumbers+actionSpace.eventVariantNumbers
        self.noVariants = len(self.variantNumbers)
        self.actionSpace = actionSpace
        self.lossFunction = lossFunction

        self.initializePopulation(initialPopSize)


        self.defaultMutateProb = 1.0/15



    def initializePopulation(self,popSize):
        self.population = [Chromosome(self.actionSpace.getRandomAction()) for i in range(popSize-1)]
        self.appendToPop(self.actionSpace.getZeroAction())

    def appendToPop(self,action):
        self.population.append(Chromosome(action))

    def generateNewPop(self,prob_Mutate=[]):
        if prob_Mutate==[]:
            prob_Mutate = self.defaultMutateProb
        losses = [self.lossFunction(chrom) for chrom in self.population]
        soFarBestChromosome = self.getBestPop()
        matePopIndices, elitePopIndices = self.selection(losses)
        self.population = self.breedPopulation(matePopIndices,elitePopIndices)
        return soFarBestChromosome

    def selection(self,losses,matePoolSize=20,eliteSize=5):
        exploitationFactor = 0.1
        probs = commonFunctions.probsFromLosses(losses,exploitationFactor)
        matePopIndices = np.random.choice(range(len(probs)), matePoolSize, replace=False, p=probs)
        elitePopIndices = np.argpartition(probs, -1*eliteSize)[-1*eliteSize:]
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
            child = crossed_child.mutateWithouCopying(self.defaultMutateProb)
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
        self.__dict__.update(action.__dict__)

    def getCopiedChrom(self):
        newChrom = Chromosome(self)
        newChrom.saveDirectly(self.activityIndizes[:],self.eventIndizes[:],self.scheduleCompressionFactors[:])
        return newChrom

    def mutate(self,mutateProb):

        newChrom = self.getCopiedChrom()
        newChrom.mutateWithouCopying(mutateProb)

        return newChrom

    def mutateWithouCopying(self,mutateProb):
        for i in range(self.actionSpace.noActivities):
            if random.random() < mutateProb:
                self.activityIndizes[i] = random.randrange(self.actionSpace.activityVariantNumbers[i])
            if random.random() < mutateProb:
                self.scheduleCompressionFactors[i] = random.uniform(0.5,1)

        for i in range(self.actionSpace.noEvents):
            if random.random() < mutateProb:
                self.eventIndizes[i] = random.randrange(self.actionSpace.eventVariantNumbers[i])

        return self


    def cross(self, chromosomeB, probB=0.5):
        newChrom = self.getCopiedChrom()

        for i in range(newChrom.actionSpace.noActivities):
            #mutate activityIndex, scheduleCompression combination together
            if random.random() < probB:
                newChrom.activityIndizes[i] = chromosomeB.activityIndizes[i]
                newChrom.scheduleCompressionFactors[i] = chromosomeB.scheduleCompressionFactors[i]


        for i in range(newChrom.actionSpace.noEvents):
            if random.random() < probB:
                newChrom.eventIndizes[i] = chromosomeB.activityIndizes[i]

        return newChrom







