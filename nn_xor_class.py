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
    def sigmoid(self, Z, derivative = None):
        if derivative is not None:
            return Z * (1 - Z)
        else:
            return 1 / (1 + np.exp(-Z))

    def tanH(self, Z, derivative = None):
        if derivative is not None:
            return 1 - np.square(Z)
        else:
            return np.tanh(Z)

    def relu(self, Z, derivative = None):
        if  derivative is not None:
            return (Z > 0).astype(float)
        else:
            return np.maximum(0, Z)

    def leakyRelu(self, Z, derivative = None):
        if derivative is not None:
            return np.clip(Z > 0, 0.01, 1.0)
        else:
            return np.where(Z > 0, Z, Z * 0.01)

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

        # Create weights and biais for each hidden layers
        for i in range(1, self.numLayers):
            parameters['W' + str(i)] = np.random.randn(self.numUnits[i], self.numUnits[i-1])
            parameters['b' + str(i)] = np.ones((self.numUnits[i], 1))

        # Create weight and biais for the final output layer
        parameters['W' + str(self.numLayers)] = np.random.randn(1, self.numUnits[-2])
        parameters['b' + str(self.numLayers)] = np.ones((1,1))

        return parameters

    ''' First step: the forward propagation '''
    def forwardPass(self):
        cache = {}
        cache['A0'] = self.trainingSet

        # go through all the hidden layers
        for i in range(1, self.numLayers-1):
            cache['W' + str(i)] = self.parameters['W' + str(i)]
            cache['b' + str(i)] = self.parameters['b' + str(i)]
            cache['Z' + str(i)] = self.preActivation(cache['W' + str(i)], cache['A' + str(i-1)], cache['b' + str(i)])
            cache['A' + str(i)] = getattr(self, self.activation)(cache['Z' + str(i)])

        # output layer
        cache['W' + str(self.numLayers-1)] = self.parameters['W' + str(self.numLayers-1)]
        cache['b' + str(self.numLayers-1)] = self.parameters['b' + str(self.numLayers-1)]
        cache['Z' + str(self.numLayers-1)] = self.preActivation(self.parameters['W' + str(self.numLayers-1)],
                                                cache['A' + str(self.numLayers-2)],
                                                self.parameters['b' + str(self.numLayers-1)])
        cache['A' + str(self.numLayers-1)] = self.sigmoid(cache['Z' + str(self.numLayers-1)])

        return cache

    ''' Second step: the backward propagation '''
    def backwardPass(self, cache):
        grads = {}

        # Gradients for the output layer
        lastLayer = str(self.numLayers-1)
        grads['dA' + lastLayer] = - (np.divide(self.trainingLabels, cache['A' + lastLayer]) - np.divide(1 - self.trainingLabels, 1 - cache['A' + lastLayer]))
        grads['dZ' + lastLayer] = cache['A' + lastLayer] - self.trainingLabels
        grads['dW' + lastLayer] = np.dot(grads['dZ' + lastLayer], cache['A' + str(int(lastLayer) - 1)].T) / self.trainingSize
        grads['db' + lastLayer] = np.sum(grads['dZ' + lastLayer], axis = 1, keepdims = True) / self.trainingSize

        # Gradients for the rest of the hidden layers
        for i in range(self.numLayers - 2, 0, -1):
            grads['dA' + str(i)] = np.dot(cache['W' + str(i+1)].T,grads['dZ' + str(i+1)])
            grads['dZ' + str(i)] = np.dot(cache['W' + str(i+1)].T,grads['dZ' + str(i+1)]) * getattr(self, self.activation)(cache['A' + str(i)], grads['dA' + str(i)])
            grads['dW' + str(i)] = np.dot(grads['dZ' + str(i)], cache['A' + str(i-1)].T) / self.trainingSize
            grads['db' + str(i)] = np.sum(grads['dZ' + str(i)], axis=1, keepdims=True) / self.trainingSize

        return grads

    ''' Update our parameters with optimized datas '''
    def updateParameters(self, cache, grads, learningRate):

        optimized = {}

        for i in range(1, self.numLayers):
            optimized['W' + str(i)] = cache['W' + str(i)] - learningRate * grads['dW' + str(i)]
            optimized['b' + str(i)] = cache['b' + str(i)] - learningRate * grads['db' + str(i)]

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
            cost = self.computeCost(self.cache['A' + str(self.numLayers - 1)])

            # print the cost
            if i % 1000 == 0:
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
            predictions = np.where(self.cache['A' + str(self.numLayers-1)] > 0.5, 1., 0.)
            acc = float((np.dot(self.trainingLabels, predictions.T) + np.dot(1 - self.trainingLabels, 1 - predictions.T)))
            acc /= float(self.trainingLabels.size)
            acc *= 100
            print("Accuracy on the training Set: " + str(acc) + "%")

