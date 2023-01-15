#test

import numpy as np
import math

inputs = [[1, 2, 3, 2.5],
        [2., 5., -1., 2],
        [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


weights2 = [[0.1, -0.14, 0.5],
[-0.5, 0.12, -0.33],
[-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

output1=np.dot(inputs, np.array(weights).T)+biases
output2=np.dot(output1, np.array(weights2).T)+biases2
print(output2)


class Layer_Dense():
        def __init__(self, n_inputs, n_neurons):
                # this can also be overwritten to load wights from before
                self.weights=0.01*np.random.randn(n_inputs, n_neurons) # its flipped to skip the transpose
                self.biases=np.zeros((1, n_neurons))
        def forward(self, inputs):
                self.output = np.dot(inputs, self.weights) + self.biases

dense1=Layer_Dense(4,1)
dense1.forward(inputs)
print(dense1.output)











