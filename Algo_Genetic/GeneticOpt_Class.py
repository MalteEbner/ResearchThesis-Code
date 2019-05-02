import random
import numpy as np
from Meta_Model import commonFunctions

class GeneticOpt():

    def __init__(self,activityVariantNumbers,lossFunction,initialPop=[]):

        self.activityVariantNumbers = activityVariantNumbers
        self.noActvities = len(activityVariantNumbers)

        if initialPop==[]:
            initialPop = self.initialPopulation(200)
        self.population = initialPop

        self.lossFunction = lossFunction

        self.defaultMutateProb = 1/15

    def mutate(self,chromosome,prob=[]):
        if prob == []:
            prob = self.defaultMutateProb
        for i in range(self.noActvities):
            if random.random() < prob:
                chromosome[i] = random.randrange(self.activityVariantNumbers[i])
        return chromosome

    def cross(self, chromosomeA, chromosomeB, probB=0.5):
        for i in range(self.noActvities):
            if random.random() < probB:
                chromosomeA[i] = chromosomeB[i]
        return chromosomeA

    def randomChromosome(self):
        chrom = []
        for i in range(self.noActvities):
            variant_ID = random.randrange(self.activityVariantNumbers[i])
            chrom.append(variant_ID)
        return chrom

    def initialPopulation(self,popSize):
        return [self.randomChromosome() for i in range(popSize)]

    def generateNewPop(self,prob_Mutate=[]):
        if prob_Mutate==[]:
            prob_Mutate = self.defaultMutateProb
        losses = [self.lossFunction(chrom) for chrom in self.population]
        soFarBestPop = self.getBestPop()[:]
        matePopIndices, elitePopIndices = self.selection(losses)
        self.population = self.breedPopulation(matePopIndices,elitePopIndices)
        return soFarBestPop


    def selection(self,losses,matePoolSize=20,eliteSize=5):
        probs = commonFunctions.probsFromLosses(losses,0.1)
        matePopIndices = np.random.choice(range(len(probs)), matePoolSize, replace=False, p=probs)
        elitePopIndices = np.argpartition(probs, -1*eliteSize)[-1*eliteSize:]
        matePopIndices = set(matePopIndices)-set(elitePopIndices)
        return matePopIndices,elitePopIndices

    def breedPopulation(self,matePopIndices, elitePopIndices):
        elitePop = [self.population[i] for i in elitePopIndices]
        matePop = [self.population[i] for i in matePopIndices]
        totalPop = elitePop[:]
        totalPop.extend(matePop)
        childPop = elitePop[:]
        for elite in elitePop:
            mutated_elite = self.mutate(elite[:])
            if True: #not mutated_elite in childPop:
                childPop.append(mutated_elite)
        while len(childPop) < len(self.population):
            mateAindex = random.randrange(len(totalPop))
            mateBindex = random.randrange(len(totalPop))
            mateA = totalPop[mateAindex]
            mateB = totalPop[mateBindex]
            crossed_mate = self.cross(mateA[:],mateB[:])
            mutated_mate = self.mutate(crossed_mate[:])
            if True:#not mutated_mate in childPop:
                childPop.append(mutated_mate)
        return childPop

    def getBestPop(self,losses=[],lossFunction=[]):
        if lossFunction==[]:
            lossFunction = self.lossFunction
        if losses==[]:
            losses = [lossFunction(chrom) for chrom in self.population]
        bestIndex = np.argmin(losses)
        return self.population[bestIndex]


