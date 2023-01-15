# basically the optimization was already contained in the backprop function in the gradients script
# but it is better to externalize this function to perform it layerwise separaetly
import numpy as np


# this just feeds back whatever improves the learning by a fraction
# this has the problem that it can easily get struck in local max/min
# also it needs to be finetuned during the process, when the steprate is too small, big
class optimizer_SDG:
    def __init__(self, learning_rate=1.0):
        self.learning_rate=learning_rate

    def update_params(self, layer):
        layer.weights +=-self.learning_rate*layer.dweights
        layer.biases +=-self.learning_rate*layer.dbiases

# to deal with said problem one could  create learning rate decay
class optimizer_SDG:
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate=learning_rate
        self.decay=decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *(1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):
        layer.weights +=-self.current_learning_rate*layer.dweights
        layer.biases +=-self.current_learning_rate*layer.dbiases
# to deal with said problem one could  create learning rate decay

    def post_update_params(self):
        self.iterations += 1


# to not get stuck in local max/min one can also implement momentum
#  this takes into account all previous updates by a fraction
# this can easily be added and parameterizied into the SDG_class
# this takes all previous updates into account
class optimizer_SDG:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate=learning_rate
        self.decay=decay
        self.iterations = 0
        self.momentum=momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *(1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
# If there is no momentum array for weights
# The array doesn't exist for biases yet either.
            layer.bias_momentums = np.zeros_like(layer.biases)
# Build weight updates with momentum - take previous
# updates multiplied by retain factor and update with
# current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
# Build bias updates
            bias_updates = \
            self.momentum * layer.bias_momentums - \
            self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
# Update weights and biases using either
# vanilla or momentum updates
            layer.weights += weight_updates
            layer.biases += bias_updates


    def post_update_params(self):
        self.iterations += 1



############################################
# however, a major insight here is that SGD is using a shared learning for all nodes
# this doesnt allways make sense, since some have greater or smaller leverage on the model
# this coud mean that updates are great for some, but then overshadow other neurons
# ada grad bypasses this isuue by offering per node/deature updates

# adagrad makes smaller updates the bigger previous steps were
# it works very simply by saving the last updates and exponentiating them
# the result is the original update by a fraction of this exponentiation
# this way it makes sure that big changes are leveled out
# it looks a bit like l2 regulaization in reverse
# 
class Optimizer_AdaGrad():
    def __init__(self, learning_rate=0.1, decay=0., epsilon=1e-7):  
    # this epsilon thing is to avoid zero division error
    # its very small, so ts effect is negligable

        self.learning_rate=learning_rate
        self.current_learning_rate = learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.iterations=0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate =self.learning_rate*(1/1*self.decay*self.iterations)

    def update_params(self,layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.biases)
        
        layer.weights += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights /(np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate *layer.dbiases /(np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1




# rmsprop works just like adagrad, as in providing feature wise learning rate adaption
# its calculated a little bit differently with a moveable cache providing momentum per feature

class Optimizer_RMSprop():

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
        rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self, layer):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))

    # all weights are added to the cache, however bgger ones move the
    #  average to be smaller, smaller ones to be bigger 
        layer.weight_cache = self.rho * layer.weight_cache + \
        (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
        (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations+=1





