# basically the optimization was already contained in the backprop function in the gradients script
# but it is better to externalize this function layerwise
import numpy as np

class optimizer_SDG:
    def __init__(self, learning_rate=1.0):
        self.learning_rate=learning_rate

    def update_params(self, layer):
        layer.weights +=-self.learning_rate*layer.dweights
        layer.biases +=-self.learning_rate*layer.dbiases

