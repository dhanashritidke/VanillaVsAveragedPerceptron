from read import read
from collections import Counter
from collections import defaultdict
import json
import math
import sys

class PerceptronLearn:
    def __init__(self, trainModelFile):
        self.trainModelFile=trainModelFile
        self.responseClasses=['Pos','Neg']
        self.validityClasses=['True','Fake']
        self.data={}
        self.trainPerceptron = []
        self.features = []
        self.featureCount=0
        self.responseWeights = {}
        self.validityWeights = {}
        self.responseBias = 0
        self.validityBias = 0
        self.averagedResponseWeights = {}
        self.averagedValidityWeights = {}
        self.averagedResponseBias = 0
        self.averagedValidityBias = 0
        self.vanillaOutputFileName = "vanillamodel.txt"
        self.averagedOutputFileName = "averagedmodel.txt"
        self.featureCounts=[]

    def readTrainingData(self):
        self.data=read(self.trainModelFile)

    def removePunctuation(self, line):
        line = line.replace("...", " ")
        punctuations = ['.', ',', '"', ';', '/', '!', "'s", '$', '-', '?', ':', '(', ')', '\n']
        for p in punctuations:
            if p == '$':
                line = line.replace(p, "$ ")
            elif p == '-' or p == '/' or p == '.':
                line = line.replace(p, " ")
            else:
                line = line.replace(p, "")
        line = line.replace("  ", " ")
        #line = line.replace("  ", " ")
        return line

    def prepareCountsForModel(self):
        stopWords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                         'its', 'on', 'of', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'this'])
        for entry in self.data:
            trainDataRow={}
            for key,value in entry.items():
                no_punctuations = self.removePunctuation(value[2].lower().strip()).split(" ")
                tokens = filter(lambda x: x not in stopWords, no_punctuations)
                featureCount= Counter(tokens)
                self.featureCounts.append(featureCount)
                if value[1] == self.responseClasses[0]:
                    trainDataRow['PositiveNegativeClass']=1
                else:
                    trainDataRow['PositiveNegativeClass'] = -1

                if value[0] == self.validityClasses[0]:
                    trainDataRow['TrueFakeClass'] = 1
                else:
                    trainDataRow['TrueFakeClass'] = -1
                trainDataRow['features']={}
                for word in featureCount:
                    if word != " " and word != '' and featureCount[word]>1 and featureCount[word]<6:
                        trainDataRow['features'][word]=featureCount[word]
            self.trainPerceptron.append(trainDataRow)

    def computeVanillaModelParameters(self):
        previousResponseWeights={}
        previousValidityWeights = {}

        previousResponseWeights = defaultdict(lambda: 0, previousResponseWeights)
        previousValidityWeights = defaultdict(lambda: 0, previousValidityWeights)

        previousResponseBias=0
        previousValidityBias=0

        epoch = 0
        while epoch < 16:
            for row in self.trainPerceptron:
                y1= row['PositiveNegativeClass']
                y2= row['TrueFakeClass']
                features= row['features']
                responseActivationValue = previousResponseBias
                validityActivationValue = previousValidityBias
                for key in features.keys():
                    responseActivationValue+=features[key]*previousResponseWeights[key]
                    validityActivationValue += features[key] * previousValidityWeights[key]

                if(responseActivationValue*y1 <= 0):
                    for key in features.keys():
                        self.responseWeights[key]= previousResponseWeights[key]+y1*features[key]
                        previousResponseWeights[key] = self.responseWeights[key]
                    self.responseBias = previousResponseBias + y1

                if(validityActivationValue*y2 <= 0):
                    for key in features.keys():
                        self.validityWeights[key]= previousValidityWeights[key]+y2*features[key]
                        previousValidityWeights[key] = self.validityWeights[key]
                    self.validityBias = previousValidityBias + y2

                previousResponseBias=self.responseBias
                previousValidityBias=self.validityBias
            epoch += 1

        print("response Bias : "+str(self.responseBias))
        print("validity Bias : " + str(self.validityBias))

    def computeAveragedModelParameters(self):
        cachedResponseWeights = {}
        cachedValidityWeights = {}

        cachedResponseWeights = defaultdict(lambda: 0, cachedResponseWeights)
        cachedValidityWeights = defaultdict(lambda: 0, cachedValidityWeights)

        cachedResponseBias = 0
        cachedValidityBias = 0

        counter = 1

        self.averagedResponseWeights = defaultdict(lambda: 0, self.averagedResponseWeights)
        self.averagedValidityWeights = defaultdict(lambda: 0, self.averagedValidityWeights)

        epoch = 0
        while epoch < 15:
            for row in self.trainPerceptron:
                y1 = row['PositiveNegativeClass']
                y2 = row['TrueFakeClass']
                features = row['features']

                responseWeightFeatureCountProduct=0
                validityWeightFeatureCountProduct=0

                for key in features.keys():
                    responseWeightFeatureCountProduct += features[key]*self.averagedResponseWeights[key]
                    validityWeightFeatureCountProduct += features[key]*self.averagedValidityWeights[key]

                if (y1*(responseWeightFeatureCountProduct+self.averagedResponseBias) <= 0):
                    for key in features.keys():
                        self.averagedResponseWeights[key] += y1*features[key]
                        cachedResponseWeights[key] += y1*features[key]*counter
                    cachedResponseBias += y1*counter
                    self.averagedResponseBias += y1


                if (y2*(validityWeightFeatureCountProduct+self.averagedValidityBias) <= 0):
                    for key in features.keys():
                        self.averagedValidityWeights[key] += y2*features[key]
                        cachedValidityWeights[key] += y2*features[key]*counter
                    cachedValidityBias += y2*counter
                    self.averagedValidityBias += y2

                counter+=1
            epoch += 1

        self.averagedValidityBias -= cachedValidityBias/counter
        self.averagedResponseBias -= cachedResponseBias/counter

        for key in cachedResponseWeights.keys():
            self.averagedResponseWeights[key] -= cachedResponseWeights[key]/counter

        for key in cachedValidityWeights.keys():
            self.averagedValidityWeights[key] -= cachedValidityWeights[key]/counter



    def writeModelToFile(self):
        self.perceptModel={}
        self.averagedPerceptModel={}

        self.perceptModel['responseWeights']=self.responseWeights
        self.perceptModel['validityWeights']=self.validityWeights
        self.perceptModel['responseBias']=self.responseBias
        self.perceptModel['validityBias']=self.validityBias

        self.averagedPerceptModel['responseWeights'] = self.averagedResponseWeights
        self.averagedPerceptModel['validityWeights'] = self.averagedValidityWeights
        self.averagedPerceptModel['responseBias'] = self.averagedResponseBias
        self.averagedPerceptModel['validityBias'] = self.averagedValidityBias

        with open(self.vanillaOutputFileName, 'w') as file:
            file.write(json.dumps(self.perceptModel))
        with open(self.averagedOutputFileName, 'w') as averageFile:
            averageFile.write(json.dumps(self.averagedPerceptModel))

        with open("featureCounts.txt", 'w') as averageFile:
            averageFile.write(json.dumps(self.featureCounts))

def runPerceptronLearn():
    perceptLearn = PerceptronLearn('./data/train-labeled.txt')
    #perceptLearn = PerceptronLearn(sys.argv[1])
    perceptLearn.readTrainingData()
    perceptLearn.prepareCountsForModel()
    perceptLearn.computeVanillaModelParameters()
    perceptLearn.computeAveragedModelParameters()
    perceptLearn.writeModelToFile()

runPerceptronLearn()

