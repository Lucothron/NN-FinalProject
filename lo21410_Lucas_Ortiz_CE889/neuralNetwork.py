from io import IncrementalNewlineDecoder
from os import error
import random
import math

# #################################### NEURON CLASS ####################################

class Neuron:

    # ***************** Initialize neuron with defined number of weights *****************

    def __init__(self, weightNum, index):
        self.AV = 0
        self.weight = []
        self.weightSum = 0
        self.weightNum = weightNum
        self.index = index
        if weightNum == -1:
            self.weight.append(1)
            self.AV = 1
        else:
            for i in range(self.weightNum):
                self.weight.append(random.random())
        # print(" --- WEIGHT --- ")
        # print(self.weight)

    # ***************** Generate activation value using sigmoid function *****************

    def actFunction(self, x):  
        self.AV = 1 / (1 + math.e ** -x)

        #print(self.AV)

    # ***************** Multiplied weight of desired layer and activation value *****************

    def weightMult(self, layer):
        weightSum = 0
        for i in range(len(layer)):
            neuron = layer[i]
            weightSum += neuron.AV * neuron.weight[self.index] 
        self.AV = 1 / (1 + math.e ** -weightSum)

    # ***************** Modify weight of specific connection ***************** 

    def updateWeight(self, newWeight, index):
        self.weight[index] += newWeight

# #################################### NEURAL NETWORK CLASS ####################################

class NeuralNetwork:

    # ***************** Initialize network *****************

    def __init__(self):
        self.inNeuronNum = 2 #input("Number of input neurons ")
        self.hidNeuronNum = 8 #input("Number of hidden neurons ")
        self.outNeuronNum = 2 #input("Number of output neurons ")

        self.inLayer = []
        self.hidLayer = []
        self.outLayer = []
        self.testDataDL = []
        self.validateDataDL = []
        self.errors = [0] * self.outNeuronNum
        self.errorSum = 0
        self.errorGrowth = 0
        self.trainError = []
        self.trainErrorSum = 0
        self.validateError = []
        self.validateErrorSum = 0
        self.minError = 5
        self.valErrorSum = 0
        self.localCounter = 0
        self.globalCounter = 0
        self.lamda = 0.8
        self.etha = 0.4
        self.alpha = 0.2
        self.trainTime = 15000
        self.validateTime = 3000
        self.hidGradients = [0] * self.hidNeuronNum
        self.hidDeltas = [0] * (self.hidNeuronNum * (self.inNeuronNum + 1))
        self.outGradients = [0] * self.outNeuronNum
        self.outDeltas = [0] * (self.outNeuronNum * (self.hidNeuronNum + 1))

        # ---- Create neurons in layer lists ----

        for i in range(int(self.inNeuronNum + 1)):
            self.inLayer.append(Neuron(int(self.hidNeuronNum), i))          # input layer + bias
        self.inLayer[-1].AV = 1

        for j in range(int(self.hidNeuronNum + 1)):
            self.hidLayer.append(Neuron(int(self.outNeuronNum), j))    # hidden layer + bias 
        self.hidLayer[-1].AV = 1

        for k in range(int(self.outNeuronNum)):                     
            self.outLayer.append(Neuron(-1, k))                   # output layer

    # ***************** Run *****************

    def run(self, trainData, validateData):
        self.readFile(trainData, True)
        self.readFile(validateData, False)
        for epochs in range(150):
            print("################ EPOCH " + str(epochs) + " ################")
            print("-- Training RMSE -- // -- Validation RMSE --")

            for i in range(self.trainTime):
                thisData = self.testDataDL[i]
                self.feedForward(thisData, True)
                self.backPropagation()

            self.trainErrorSum = 0    
            
            for i in range(self.trainTime):
                thisData = self.testDataDL[i]
                self.feedForward(thisData, True)
                #print(str(self.trainErrorSum) + " --- " + str(self.validateErrorSum))

            self.errorCalc(True)

            for i in range(self.validateTime):
                thisData = self.validateDataDL[i]
                self.feedForward(thisData, False)

            self.errorCalc(False)

            print(str(self.trainError[-1]) + " // " + str(self.validateError[-1]) + "\n")

            self.trainErrorSum = 0
            self.validateErrorSum = 0

            if epochs > 3:
                saveData = open("errorGraph.csv", "a")

                saveData.write(str(epochs) + ", " + str(self.validateError[-1]))

                saveData.close()


                if self.validateError[-1] < self.minError:
                    self.minError = self.validateError[-1]
                    optimalVals = open("optimalValues.csv", "w")

                    for i in range(self.inNeuronNum + 1):
                        neuron = self.inLayer[i]
                        for j in range(self.hidNeuronNum):
                            optimalVals.write(str(neuron.weight[j]) + ", ")
                        optimalVals.write("\n")

                    for i in range(self.hidNeuronNum + 1):
                        neuron = self.hidLayer[i]
                        for j in range(self.outNeuronNum):
                            optimalVals.write(str(neuron.weight[j]) + ", ")
                        optimalVals.write("\n")
                    
                    optimalVals.close()

                if self.validateError[-1] < 0.1:
                    print("--- Error below 0.1 ---")
                    break
            
                if self.validateError[-1] > self.validateError[-2]:
                    self.errorGrowth += 1
                else:
                    self.errorGrowth = 0

                if self.errorGrowth >= 5:
                    print("--- Continuous error growth ---")
                    break

            random.shuffle(self.testDataDL)
            random.shuffle(self.validateDataDL)
            
    # ***************** Read data from csv file *****************

    def readFile(self, fileName, fileType):
        if fileType:
            testData = open(fileName, "r", encoding="utf-8-sig")
            for line in testData:
                self.testDataDL.append(line.rstrip().split(","))
            testData.close
        else:
            validateData = open(fileName, "r", encoding="utf-8-sig")
            for line in validateData:
                self.validateDataDL.append(line.rstrip().split(","))
            validateData.close()

    # ***************** Error calculation *****************

    def errorCalc(self, train):
        if train:
            self.trainError.append(math.sqrt((self.trainErrorSum) / self.trainTime))
        else:
            self.validateError.append(math.sqrt((self.validateErrorSum) / self.validateTime))

    # ***************** Feed forward *****************

    def feedForward(self, thisData, train):
        # print("--- DATA ---")
        # print(thisData)

        # ---- In layer activation values ----

        for i in range(self.inNeuronNum):
            neuron = self.inLayer[i]
            neuron.AV = float(thisData[i])

        # ---- In layer to hidden layer ----

        for i in range(self.hidNeuronNum):
            self.hidLayer[i].weightMult(self.inLayer)

        # ---- Hidden layer to output layer ----

        for i in range(self.outNeuronNum):
            self.outLayer[i].weightMult(self.hidLayer)

        # ---- Calculate error in each out neuron ---- 

        errorTemp = 0
        for i in range(self.outNeuronNum):
            neuron = self.outLayer[i]
            self.errors[i] = (float(thisData[i + 2]) - neuron.AV)
            errorTemp += pow(self.errors[i], 2)
        
        if train:
            self.trainErrorSum += (errorTemp / 2)
        else:
            self.validateErrorSum += (errorTemp / 2)

        # print("--- ERROR CALCULATIONS ---")
        # print(self.errors)

    # ***************** Back propagation *****************
    
    def backPropagation(self):

        # ---- Hidden layer ----

        for i in range(self.outNeuronNum):
            neuron = self.outLayer[i]
            self.outGradients[i] = (self.lamda * neuron.AV * (1 - neuron.AV) * self.errors[i])

        for i in range(self.hidNeuronNum + 1):
            neuron = self.hidLayer[i]
            for j in range(self.outNeuronNum):
                momentum = self.alpha * self.outDeltas[(i * self.outNeuronNum) + j]
                self.outDeltas[(i * self.outNeuronNum) + j] = (self.etha * self.outGradients[j] * neuron.AV + momentum)

        # ---- Input layer ----

        for i in range(self.hidNeuronNum):
            neuron = self.hidLayer[i]
            tempGradSum = 0
            for j in range(self.outNeuronNum):
                tempGradSum += neuron.weight[j] * self.outGradients[j]
            self.hidGradients[i] = (self.lamda * neuron.AV * (1 - neuron.AV) * (tempGradSum))

        for i in range(self.inNeuronNum):
            neuron = self.inLayer[i]
            for j in range(self.hidNeuronNum):
                momentum = self.alpha * self.hidDeltas[(i * self.hidNeuronNum) + j]
                self.hidDeltas[(i * (self.hidNeuronNum - 1)) + j] = (self.etha * self.hidGradients[j] * neuron.AV + momentum)
        
        for i in range(self.hidNeuronNum + 1):
            neuron = self.hidLayer[i]
            for j in range(self.outNeuronNum):
                neuron.updateWeight(self.outDeltas[i * self.outNeuronNum + j], j)

        for i in range(self.inNeuronNum + 1):
            neuron = self.inLayer[i]
            for j in range(self.hidNeuronNum):
                neuron.updateWeight(self.hidDeltas[i * self.hidNeuronNum + j], j)

        # for neuron in self.inLayer:
        #     print("in neuron weight --- " + str(neuron.weight))

        # for neuron in self.hidLayer:
        #     print("hid neuron weight --- " + str(neuron.weight))

        # for neuron in self.outLayer:
        #     print("out neuron weight --- " + str(neuron.weight))

    def loadValues(self):
        optValues = open("optimalValues.csv", "r", encoding = "utf-8-sig")
        tempVal = []

        for line in optValues:
            tempVal.append(line.rstrip().split(","))
        optValues.close()

        for i in range(self.inNeuronNum + 1):
            neuron = self.inLayer[i]
            for j in range(self.hidNeuronNum):
                neuron.weight[j] = tempVal[i][j]

        for i in range(self.hidNeuronNum + 1):
            neuron = self.hidLayer[i]
            for j in range(self.outNeuronNum):
                neuron.weight[j] = tempVal[i + (self.inNeuronNum + 1)][j]

        for neuron in self.inLayer:
            print("in neuron weight --- " + str(neuron.weight))

        for neuron in self.hidLayer:
            print("hid neuron weight --- " + str(neuron.weight))

        for neuron in self.outLayer:
            print("out neuron weight --- " + str(neuron.weight))