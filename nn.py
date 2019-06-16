""" L-Layer Neural Network on XOR dataset """
import argparse
from nn_xor_class import NeuralNetwork

""" Arguments of the script """
parser = argparse.ArgumentParser(description='Train a L-Layer Neural Network on XOR dataset with various activation functions.')

# Number of layers
parser.add_argument('-l', "--layers", type=int,
                    help="Number of layers in your NN (including input and output layers).  Default: 5.",
                    action="store", default=5)
# Number of units for each layer
parser.add_argument('-u', '--units',
                    help="Number of units in each hidden layer separated by a comma (excluding input and output layers). Default: 4,3,2.",
                    action="store", default="4,3,2")

# Size of the dataset
parser.add_argument('-s', '--size',
                    help="How many examples should be generated in our training set. Default: 5000.",
                    action="store", type=int, default=5000)

# Define our number of iterations
parser.add_argument('-i', '--iterations',
                    help="Choose the number of iterations we want. Default: 10000.",
                    action="store", type=int, default=10000)

# Define our learning Rate
parser.add_argument('-r', '--learning-rate',
                    help="Pick a Learning rate for your neural Network. Default: 0.5.",
                    action="store", type=float, default=0.5)

# Activation function for hidden layers (minus the output layer)
parser.add_argument('-a', '--activation',
                    help="Activation function for your hidden layers. The output layer will always be a sigmoid. Default: sigmoid.",
                    action="store", choices=["sigmoid", "tanH", "relu", "leakyRelu"], default="sigmoid")

args = parser.parse_args()
NN = NeuralNetwork(args.layers, args.units.split(','), args.activation)
NN.showConfig(args)
NN.createXorDataset(args.size)
NN.train(iterations = args.iterations, learningRate = args.learning_rate)
NN.testPrediction()