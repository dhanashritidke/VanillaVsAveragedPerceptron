from collections import Counter
from read import read
import json
import sys

class PerceptronClassifier:
    def __init__(self, modelFile, file1):
        self.tokens = {}
        self.totalWords = 0
        self.modelParametersFile = modelFile
        self.model={}
        self.testFile=file1
        self.outputFileName = "percepoutput.txt"
        self.responseClasses = ['Pos', 'Neg']
        self.validityClasses = ['True', 'Fake']
        self.validity=''
        self.response=''

    def removePunctuation(self, line):
        punctuations = ['.', ',', '"', ';', '/', '!', "'s", '$', '-']
        for p in punctuations:
            if p == '$':
                line = line.replace(p, "$ ")
            elif p == '-' or p == '/':
                line = line.replace(p, " ")
            else:
                line = line.replace(p, "")
        return line

    def readModelParameters(self):
        with open(self.modelParametersFile, 'r') as fp:
            self.model = json.load(fp)

    def classify(self):
        txt_file = open(self.testFile, "r")
        outputFile = open(self.outputFileName, "w")
        for line in txt_file:
            line = line.strip()
            review_id = line.split(' ', 1)[0]
            reviewString = line.split(' ', 1)[1]
            featureCount = Counter(self.removePunctuation(reviewString.lower().strip()).split(" "))
            responseResult=self.model['responseBias']
            responseWeights=self.model['responseWeights']
            for word in featureCount:
                if word in responseWeights.keys():
                    responseResult+=featureCount[word]*responseWeights[word]
            if responseResult>0:
                self.response=self.responseClasses[0]
            else:
                self.response = self.responseClasses[1]

            validityResult = self.model['validityBias']
            validityWeights = self.model['validityWeights']
            for word in featureCount:
                if word in validityWeights.keys():
                    validityResult += featureCount[word] * validityWeights[word]
            if validityResult > 0:
                self.validity = self.validityClasses[0]
            else:
                self.validity = self.validityClasses[1]

            outputFile.write("%s %s %s\n" % (review_id, self.validity, self.response))

def percepClassifier():
    file_name = "./data/dev-text.txt"
    modelFile = "averagedmodel.txt"
    percepClassifier = PerceptronClassifier(modelFile,file_name)
    #percepClassifier = PerceptronClassifier(sys.argv[1],sys.argv[2])
    percepClassifier.readModelParameters()
    percepClassifier.classify()

percepClassifier()