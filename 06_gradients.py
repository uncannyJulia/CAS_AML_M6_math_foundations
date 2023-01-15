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

# but to make this work all the code needs to be integrated layerwise in the classes created
# the class architecture allows easy spread of forward and backwand operations
# to avoid long formulas it is best to also call it layerwise - so every layer separately delivers its forward and backward results
# then these reults can be passed up/ down to the next one

# so far i was easy because the dervatives of relu and dense are easy to take. 
# One big issue might be the derivatices of sigmoid/ cross entropy. 
# I will just implement an eccepted formula, 
# because I lack the skills to express the mathematical explanation as to why this is the derivative.


# cross-entropy

def CCE_loss_prop(dvalues, y_true):
    samples=len(dvalues)            # the result needs to be normalized by how many outputs there are
    labels=len(dvalues[0])
    if len(y_true.shape) == 1:
        y_true = np.eye(labels)[y_true]
    dinputs = -y_true / dvalues
    dinputs = dinputs / samples



def SM_prop(dvalues, output):                   # output is the result from the forward pass
    dinputs = np.empty_like(dvalues)
    for index, (single_output, single_dvalues) in enumerate(zip(output, dvalues)):
        # Flatten output array
        single_output = single_output.reshape(-1, 1)
        # Calculate Jacobian matrix of the output
        jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
        
        dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

