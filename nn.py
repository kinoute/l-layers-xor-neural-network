""" L-Layers Neural Network on XOR dataset """
import argparse
from nn_xor_class import NeuralNetwork

""" Arguments of the script """
parser = argparse.ArgumentParser(description='Train a L-Layers Neural Network on XOR dataset with various activation functions.')

# Number of layers
parser.add_argument('-l', "--layers", type=int,
                    help="Number of layers in your NN (including the output layer).  Default: 4.",
                    action="store", default=4)

# Number of units for each layer
parser.add_argument('-u', '--units', type=str,
                    help="Number of units in each hidden layer separated by a comma (including output layer). Default: 4,2,1.",
                    action="store", default="4,2,1")

# Size of the dataset
parser.add_argument('-s', '--size', type=int,
                    help="How many examples should be generated in our training set. Default: 1000.",
                    action="store", default=1000)

# Define our number of iterations
parser.add_argument('-i', '--iterations', type=int,
                    help="Choose the number of iterations we want. Default: 10000.",
                    action="store", default=10000)

# Define our learning Rate
parser.add_argument('-r', '--learning-rate', type=float,
                    help="Pick a Learning rate for your neural Network. Default: 1.5.",
                    action="store", default=1.5)

# Activation function for hidden layers (minus the output layer)
parser.add_argument('-a', '--activation',
                    help="Activation function for your hidden layers. The output layer will always be a sigmoid. Default: tanH.",
                    action="store", choices=["sigmoid", "tanH", "relu", "leakyRelu"], default="tanH")

args = parser.parse_args()

NN = NeuralNetwork(args.layers, args.units.split(','), args.activation)
NN.showConfig(args)
NN.createXorDataset(args.size)
NN.train(iterations = args.iterations, learningRate = args.learning_rate)
NN.testPrediction()
