import neuralNetwork
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.network = neuralNetwork.NeuralNetwork()
        self.network.loadValues()
    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        pass # this pass can be removed once you add some code
        input_row = input_row.split(',')

        xNormalize = ((float(input_row[0]) - (-775.4994493)) / (808.7342997 - (-775.4994493)))
        yNormalize = ((float(input_row[1]) - 65.17227597) / (566.3060612 - 65.17227597))

        data = [xNormalize, yNormalize, 0, 0]

        self.network.feedForward(data, False)
        
        outVal = [None] * self.network.outNeuronNum

        for i in range(self.network.outNeuronNum):
            neuron = self.network.outLayer[i]
            outVal[i] = neuron.AV

        xDenormalize = (outVal[0] * (8 - (-3.061109146)) + -3.061109146)
        yDenormalize = (outVal[1] * (5.909986576 - (-6.302121003)) + -6.302121003)

        return([xDenormalize, yDenormalize])