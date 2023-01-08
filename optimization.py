# backpropagation is performed layerwise with the chain rule
# the output of the activation is basicaly a chained function, where one takes the outhers output as input
# to assess the leverage one input has on the output one takes partial derivatives of these chained functions
# it comes down taking to multiplying the derivative(result of the previous forward pass) by the derivative of the activation
# in the case of relu its encredibly easy, because it is either 0 or 1

# lets do the outer most layer - activation
import numpy as np
# the derivatives of the activation
dvalues=np.array([[1.,1.,1.]])
weights =   np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
# since the derivative of previous function layer(x) is just the weight numbers we can just use the weights matrix
# we need to calculare it thinking for how one input travels through the system, so basically row wise not column wise
# since np.dot does row and column, we have to reverse it to have the right order

def single_sample_prop(dvalues, weights):
    dinputs = np.dot(dvalues[0], weights.T)
    print(f"simple sample propapgation outcome: {dinputs}")

single_sample_prop(dvalues, weights)

# just like in the forward pass we can repurpose the np.dot also for matrix products, we just have to keep the directions in check!
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])   # outcome of the layer

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

inputs = np.array([[1, 2, 3, 2.5],
                    [2., 5., -1., 2],
                    [-1.5, 2.7, 3.3, -0.8]])

biases = np.array([[2, 3, 0.5]])

def batch_prop(dvalues, weights, inputs, biases):
    dvalues=dvalues
    drelu = dvalues.copy()
    drelu[dvalues <= 0] = 0
    dinputs = np.dot(dvalues[0], weights.T)
    dweights = np.dot(inputs.T, dvalues)
    dbiases = np.sum(dvalues, axis=0, keepdims=True)
    weights += -0.001 * dweights
    biases += -0.001 * dbiases
    print(f"new weights are: {weights} \n new biases are {biases}")
batch_prop(dvalues, weights, inputs, biases)
