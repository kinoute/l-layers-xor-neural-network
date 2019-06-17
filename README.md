# L-Layers XOR Neural Network

A L-Layers XOR Neural Network using only Python and Numpy that learns to predict the XOR logic gates.

## Script

The script was initially made as a `.ipynb` jupyter file and later refactored as a class and a script where arguments can be passed to the neural network.

There are two files:

* `nn.py`: the main script that uses our NeuralNetwork class stored in the other file ;
* `nn_xor_class.py`: our NeuralNetwork class.

## Arguments

You can change the behavior of the Neural Network by using arguments when running the script. For example, you can change the activation function of the hidden layers, the learning rate etc. Here are the arguments allowed when running `nn.py`. All arguments are optional.

### Usage

```
python nn.py [-h] [-l LAYERS] [-u UNITS] [-s SIZE] [-i ITERATIONS] [-r LEARNING_RATE]
                    [-a {sigmoid,tanH,relu,leakyRelu}]

  -h, --help            show this help message and exit

  -l LAYERS, --layers LAYERS
                        Number of layers in your NN (including input and
                        output layers). Default: 5.

  -u UNITS, --units UNITS
                        Number of units in each hidden layer separated by a
                        comma (excluding input and output layers). Default:
                        4,3,2.

  -s SIZE, --size SIZE  How many examples should be generated in our training
                        set. Default: 5000.

  -i ITERATIONS, --iterations ITERATIONS
                        Choose the number of iterations we want. Default: 10000.

  -r LEARNING_RATE, --learning-rate LEARNING_RATE
                        Pick a Learning rate for your neural Network. Default: 1.5.

  -a {sigmoid,tanH,relu,leakyRelu}, --activation {sigmoid,tanH,relu,leakyRelu}
                        Activation function for your hidden layers. The output
                        layer will always be a sigmoid. Default: "sigmoid".

```
