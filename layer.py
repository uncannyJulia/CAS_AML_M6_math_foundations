import numpy as np
import math

def simple_layer(inputs:np.ndarray, weights: np.ndarray, biases:np.ndarray):
    num_neurons=weights.shape[0]
    num_rows=weights.shape[1]
    output_list=[]
    
    for i in range(0,num_neurons,1):
        weight_row=weights[i]
        row_list=[]
        for i, val in enumerate(inputs):
            row=inputs[i]*weight_row[i]
            row_list.append(row)
            term=sum(row_list)
        output = term
        output_list.append(output)
    
    result=output_list + biases
    print(result)
    
inputs = [8, 2, 7, 2.5]
weights= np.array([ [0.1, 0.2, 0.3, 1],
                  [0.5, -3, 4, -0.125],
                  [-0.45, -0.44, 0.8, 2.77] ])
biases=np.array([2, 3, 0.5])

simple_layer(inputs=inputs, weights=weights, biases=biases)


def better_layer(inputs: list, weights:np.ndarray , biases=np.array):
    output=[]
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            #print(n_input, weight)
            neuron_output += n_input*weight
        neuron_output += neuron_bias
        output.append(neuron_output)
    print(output)

better_layer(inputs=inputs, weights=weights, biases=biases)

def numpy_layer(inputs: list, weights:np.ndarray , biases=np.array):
    outputs=np.dot(weights, inputs) + biases
    print(outputs)
    return outputs

numpy_layer(inputs=inputs, weights=weights, biases=biases)