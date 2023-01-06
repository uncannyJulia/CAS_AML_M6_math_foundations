import numpy as np

######## relu
def relu(input: list):
    output=[]
    for i in input:
        output.append(max(0,i))
    return(output)


class activation_relu:
    def forward(self, input):
        self.output=np.maximum(0, input)


######## softmax, normalizes confodence scores of outputs between 0,1

#formula=(e**zi,j)/sum(e**zi,l) 

#unnormalized outputs
layer_outputs = [4.8, 1.21, 2.385]
E = 2.71828182846
#exponentiated
exp=[]
for output in layer_outputs:
    exp.append(E**output)
norm_base=sum(exp)
norm_values=[]
for value in exp:
    norm_values.append(value/norm_base)
print(norm_values)

class activation_softmax:
    def forward(self, inputs):
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probs=exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output=probs





