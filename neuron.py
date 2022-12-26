def neuron(inputs: list, weights: list, bias: float):
    row_list=[]
    for i, val in enumerate(inputs):
        row=inputs[i]*weights[i]
        row_list.append(row)
    term=sum(row_list)
    output = term + bias
    print(output)
    return output

inputs=[1,2,3]
weights=[0.2, 0.8, -0.5]
bias= 2.0

neuron(inputs=inputs, weights=weights, bias=bias)
