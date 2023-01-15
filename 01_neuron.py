# a simple function that calculates the forward pass through a single reuron

def simple_neuron(inputs: list, weights: list, bias: float):
    row_list=[]
    for i, val in enumerate(inputs):
        row=inputs[i]*weights[i]
        row_list.append(row)
    term=sum(row_list)
    output = term + bias
    print(output)
    return output


# test
inputs=[6,8,9]
weights=[0.2, 0.5, -0.5]
bias= 1.0
simple_neuron(inputs=inputs, weights=weights, bias=bias)

import numpy as np
def numpy_neuron(inputs:list, weights:list, bias: float):
    output = np.dot(inputs, weights)+ bias
    print(output)
    return(output)

numpy_neuron(inputs=inputs, weights=weights, bias=bias)

