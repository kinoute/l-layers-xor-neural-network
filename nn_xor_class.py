import numpy as np

class NeuralNetwork(object):
    """Create a Neural Network for L Layers and bitwise XOR"""

    trainingSet = None
    trainingLabels = None
    trainingSize = None

    parameters = None
    cache = None

    ''' initialize our NN '''
    def __init__(self, numLayers, numUnits, activation):
        self.numLayers = numLayers
        self.numUnits = list(map(int, numUnits))
        self.activation = activation

        # Don't do anything if the number of hidden units doesn't match the hidden layers
        if len(self.numUnits) != self.numLayers - 2:
            raise Exception("The number of hidden units should be equal to the number of total layers minus 2 (input and output layers).")

    ''' Display the NN configuration at the beginning '''
    def showConfig(self, args):
        print('Neural Network Configuration: \n')
        print('- Number of Layers:', self.numLayers)
        print('- Hidden Units by Layer:', self.numUnits)
        print('- Activation Function:', self.activation)
        print('- Training Set Size:', args.size)
        print('- Number of Iterations:', args.iterations)
        print('- Learning Rate:', args.learning_rate, '\n')

    ''' Compute Z '''
    def preActivation(self, W, X, b):
        return np.dot(W, X) + b

    ''' Activation functions '''
    def sigmoid(self, F, derivative = False, heuristic = False):
        if derivative:
            return F * (1 - F)
        elif heuristic:
            return np.sqrt(1 / heuristic)
        else:
            return 1 / (1 + np.exp(-F))

    def tanH(self, F, derivative = False, heuristic = False):
        if derivative:
            return 1 - np.square(F)
        elif heuristic:
            return np.sqrt(1 / heuristic)
        else:
            return np.tanh(F)

    def relu(self, F, derivative = False, heuristic = False):
        if  derivative:
            return (F > 0).astype(float)
        elif heuristic:
            return np.sqrt(2 / heuristic)
        else:
            return np.maximum(0, F)

    def leakyRelu(self, F, derivative = False, heuristic = False):
        if derivative:
            return np.clip(F > 0, 0.01, 1.0)
        elif heuristic:
            return np.sqrt(2 / heuristic)
        else:
            return np.where(F > 0, F, F * 0.01)

    ''' Loss and Cost functions '''
    def computeLoss(self, A):
        return (self.trainingLabels * np.log(A) + (1 - self.trainingLabels) * np.log(1 - A))

    def computeCost(self, A):
        return - (np.sum(self.computeLoss(A)) / self.trainingSize)

    ''' Create our bitwise XOR dataset '''
    def createXorDataset(self, size):
        X1 = np.random.choice([0, 1], (size,1))
        X2 = np.random.choice([0, 1], (size,1))

        X = np.concatenate((X1, X2), axis=1)
        Y = np.bitwise_xor(X[:, 0], X[:, 1])

        self.trainingSet = X.T
        self.trainingLabels = Y
        self.trainingSize = self.trainingSet.shape[1]

        # Add the number of rows of the training set to our layers sizes list
        self.numUnits.insert(0, self.trainingSet.shape[0])

        # Add the output layer dimension at the end
        self.numUnits.append(1)

    ''' Initialize our parameters (weights and biais) '''
    def initializeParameters(self):

        parameters = {}

        # Create weights and biais for each hidden layer with heuristic
        for i in range(1, self.numLayers):
            parameters[f'W{i}'] = np.random.randn(self.numUnits[i], self.numUnits[i-1]) * getattr(self, self.activation)(None, heuristic = self.numUnits[i-1])
            parameters[f'b{i}'] = np.ones((self.numUnits[i], 1))

        # Create weight and biais for the final output layer with heuristic
        parameters[f'W{self.numLayers}'] = np.random.randn(1, self.numUnits[-2]) * self.sigmoid(None, heuristic = self.numUnits[i-1])
        parameters[f'b{self.numLayers}'] = np.ones((1,1))

        return parameters

    ''' First step: the forward propagation '''
    def forwardPass(self):
        cache = {}
        cache['A0'] = self.trainingSet

        # go through all the hidden layers
        for i in range(1, self.numLayers - 1):
            cache[f'W{i}'] = self.parameters[f'W{i}']
            cache[f'b{i}'] = self.parameters[f'b{i}']
            cache[f'Z{i}'] = self.preActivation(cache[f'W{i}'], cache[f'A{i-1}'], cache[f'b{i}'])
            cache[f'A{i}'] = getattr(self, self.activation)(cache[f'Z{i}'])

        # output layer
        cache[f'W{self.numLayers-1}'] = self.parameters[f'W{self.numLayers-1}']
        cache[f'b{self.numLayers-1}'] = self.parameters[f'b{self.numLayers-1}']
        cache[f'Z{self.numLayers-1}'] = self.preActivation(self.parameters[f'W{self.numLayers-1}'], cache[f'A{self.numLayers-2}'], self.parameters[f'b{self.numLayers-1}'])
        cache[f'A{self.numLayers-1}'] = self.sigmoid(cache[f'Z{self.numLayers-1}'])

        return cache

    ''' Second step: the backward propagation '''
    def backwardPass(self, cache):
        grads = {}

        # Gradients for the output layer
        lastLayer = self.numLayers - 1
        grads[f'dA{lastLayer}'] = - (np.divide(self.trainingLabels, cache[f'A{lastLayer}']) - np.divide(1 - self.trainingLabels, 1 - cache[f'A{lastLayer}']))
        grads[f'dZ{lastLayer}'] = cache[f'A{lastLayer}'] - self.trainingLabels # because we have a sigmoid as output
        grads[f'dW{lastLayer}'] = np.dot(grads[f'dZ{lastLayer}'], cache[f'A{lastLayer - 1}'].T) / self.trainingSize
        grads[f'db{lastLayer}'] = np.sum(grads[f'dZ{lastLayer}'], axis = 1, keepdims = True) / self.trainingSize

        # Gradients for the rest of the hidden layers
        for i in range(self.numLayers - 2, 0, -1):
            grads[f'dA{i}'] = np.dot(cache[f'W{i+1}'].T,grads[f'dZ{i+1}'])
            grads[f'dZ{i}'] = grads[f'dA{i}'] * getattr(self, self.activation)(cache[f'A{i}'], derivative = True)
            grads[f'dW{i}'] = np.dot(grads[f'dZ{i}'], cache[f'A{i-1}'].T) / self.trainingSize
            grads[f'db{i}'] = np.sum(grads[f'dZ{i}'], axis = 1, keepdims = True) / self.trainingSize

        return grads

    ''' Update our parameters with optimized datas '''
    def updateParameters(self, cache, grads, learningRate):

        optimized = {}

        for i in range(1, self.numLayers):
            optimized[f'W{i}'] = cache[f'W{i}'] - learningRate * grads[f'dW{i}']
            optimized[f'b{i}'] = cache[f'b{i}'] - learningRate * grads[f'db{i}']

        return optimized

    ''' Main model with everything to train our NN '''
    def train(self, iterations, learningRate):

        if iterations < 1000:
            raise Exception("Please pick a higher number of iterations.")

        print("Initializing the weights...")
        self.parameters = self.initializeParameters()

        print("Starting the training...")

        for i in range(1, iterations):

            # forward pass
            self.cache = self.forwardPass()

            # compute cost
            cost = self.computeCost(self.cache[f'A{self.numLayers - 1}'])

            # print the cost
            if i % 250 == 0:
                print("Cost ater", i, "iterations:", cost)

            # calculate gradients
            grads = self.backwardPass(self.cache)

            # update parameters
            self.parameters = self.updateParameters(self.cache, grads, learningRate)

        print("Final cost:", cost)

    ''' Get the accuracy of our NN on the training set '''
    def testPrediction(self):
        if self.parameters == None:
            raise Exception("You have to train your Neural Network first.")
        else:
            self.cache = self.forwardPass()
            predictions = np.where(self.cache[f'A{self.numLayers-1}'] > 0.5, 1., 0.)
            acc = float((np.dot(self.trainingLabels, predictions.T) + np.dot(1 - self.trainingLabels, 1 - predictions.T)))
            acc /= float(self.trainingLabels.size)
            acc *= 100
            print(f"Accuracy on the Training Set: {acc}%")

