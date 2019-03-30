## What is a Neural Network(NN)?

A NN is a simple mathematical function which maps inputs to a desired output along with some weights.

OR

It is basically an algorithm that learns on the pairs of examples input and output data, detects some kind of patterns, and predicts the output based on an unseen input data. 

Neural Networks consist of the following components

    An input layer, x
    An arbitrary amount of hidden layers
    An output layer, ŷ
    A set of weights and biases between each layer, W and b
    A choice of activation function for each hidden layer, σ. 
    
    In this tutorial, we’ll use a Sigmoid activation function.

# Jump to
[1. Basic NN](#basic)


<a name="basic"></a>
## 1. Basic NN explanation


<p align="center">
  <img width="299" height="340" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/399px-Colored_neural_network.svg.png">
</p>

> The circles represent neurons while the lines represent synapses. The role of a synapse is to multiply the inputs and weights. You can think of weights as the “strength” of the connection between neurons. 

> Weights primarily define the output of a neural network. However, they are highly flexible. After, an activation function is applied to return an output.

**Things that we're gonna perform -**


    1. Takes inputs as a matrix (2D array of numbers)

    2. Multiplies the input by a set weights (performs a dot product aka matrix multiplication)

    3. Applies an activation function

    4. Returns an output
        
    5. Error is calculated by taking the difference from the desired output from the data and the predicted output. This creates our gradient descent, which we can use to alter the weights

    6. The weights are then altered slightly according to the error.

    7. To train, this process is repeated 1,000+ times. The more the data is trained upon, the more accurate our outputs will be.

[Dot product](https://www.khanacademy.org/math/precalculus/precalc-matrices/multiplying-matrices-by-matrices/v/matrix-multiplication-intro) is like making *strength* in our synapse(Connection between neurons).

At its core, neural networks are simple. They just perform a dot product with the input and weights and apply an activation function. When weights are adjusted via the gradient of loss function, the network adapts to the changes to produce more accurate outputs.

**Our neural network will model a single hidden layer with three inputs and one output.
In the network, we will be predicting the score of our exam based on the inputs of how many hours we studied and how many hours we slept the day before.** 

Our test score is the output. Here’s our sample data of what we’ll be training our Neural Network on:

| Hours Studied        | Hours Slept   | Test Score  |
| :-------------: |:-------------:| :------: |
|  2      | 9 |        `91` |
|  1    | 5      |       `80`   |
|  3 |  6     |          `85` |
|  4 |  8     |          `?` |

As you may have noticed, the `?` in this case represents what we want our neural network to predict. In this case, we are predicting the test score of someone who studied for four hours and slept for eight hours based on their prior performance.


## Forward Propagation

Let’s start coding! Open up a new python file. You’ll want to import numpy as it will help us with certain calculations.

First, let’s import our data as numpy arrays using np.array. We’ll also want to normalize our units as our inputs are in hours, but our output is a test score from 0-100. Therefore, we need to scale our data by dividing by the maximum value for each variable.

```python


import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

```
Next, let’s define a python `class` and write an `init` function where we’ll specify our parameters such as the input, hidden, and output layers.

```python


class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3
```
> Note that weights are generated randomly and between 0 and 1.

### The calculations behind our network

In the data set, our input data, `X`, is a 3x2 matrix. Our output data, `y`, is a 3x1 matrix. Each element in matrix X needs to be multiplied by a corresponding weight and then added together with all the other results for each neuron in the hidden layer.


<p align="center">
  <img src="https://enlight.nyc/img/nn-calc.png">
</p>

This image breaks down what our neural network actually does to produce an output. First, the products of the random generated weights (.2, .6, .1, .8, .3, .7) on each synapse and the corresponding inputs are summed to arrive as the first values of the hidden layer. These sums are in a smaller font as they are not the final values for the hidden layer.

```python
(2 * .2) + (9 * .8) = 7.6
(2 * .6) + (9 * .3) = 3.9
(2 * .1) + (9 * .7) = 6.5
```

To get the final value for the hidden layer, we need to apply the [activation function](https://en.wikipedia.org/wiki/Activation_function). The role of an activation function is to introduce nonlinearity. An advantage of this is that the output is mapped from a range of 0 and 1, making it easier to alter weights in the future.

There are many activation functions out there. In this case, we’ll stick to one of the more popular ones - the sigmoid function.

<p align="center">
  <img src="https://enlight.nyc/img/sigmoid.png">
</p>

```python
S(7.6) = 0.999499799
S(7.5) = 1.000553084
S(6.5) = 0.998498818
```
Now, we need to use matrix multiplication again, with another set of random weights, to calculate our output layer value.
```python
(.9994 * .4) + (1.000 * .5) + (.9984 * .9) = 1.79832
```
Lastly, to normalize the output, we just apply the activation function again.
```python
S(1.79832) = .8579443067
```
And, there you go! Theoretically, with those weights, out neural network will calculate `.85` as our test score! However, our target was `.92`. Our result wasn’t poor, it just isn’t the best it can be. We just got a little lucky when I chose the random weights for this example.

How do we train our model to learn? Well, we’ll find out very soon. For now, let’s countinue coding our network.

If you are still confused, I highly reccomend you check out this informative video which explains the structure of a neural network with the same example.


### Implementing the calculations

Now, let’s generate our weights randomly using np.random.randn(). Remember, we’ll need two sets of weights. One to go from the input to the hidden layer, and the other to go from the hidden to output layer.
```python
#weights
self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
```
Once we have all the variables set up, we are ready to write our forward propagation function. Let’s pass in our input,     `X`, and in this example, we can use the variable `z` to simulate the activity between the input and output layers. As explained, we need to take a dot product of the inputs and weights, apply an activation function, take another dot product of the hidden layer and second set of weights, and lastly apply a final activation function to recieve our output:
```python
def forward(self, X):
#forward propagation through our network
  self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
  self.z2 = self.sigmoid(self.z) # activation function
  self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
  o = self.sigmoid(self.z3) # final activation function
  return o
```
Lastly, we need to define our sigmoid function:
```python
def sigmoid(self, s):
  # activation function
  return 1/(1+np.exp(-s))
```
And, there we have it! A (untrained) neural network capable of producing an output.
```python
import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

NN = Neural_Network()

#defining our output
o = NN.forward(X)

print "Predicted Output: \n" + str(o)
print "Actual Output: \n" + str(y)
```

As you may have noticed, we need to train our network to calculate more accurate results.
























