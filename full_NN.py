import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
# Remember input values
        self.inputs = inputs
# Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
# Backward pass
    def backward(self, dvalues):
# Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
# Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
# ReLU activation
class Activation_ReLU:
# Forward pass
    def forward(self, inputs):
# Remember input values
        self.inputs = inputs
# Calculate output values from inputs
        self.output = np.maximum(0, inputs)
# Backward pass
    def backward(self, dvalues):
# Since we need to modify original variable,
# let’s make a copy of values first
        self.dinputs = dvalues.copy()
# Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
# Softmax activation
class Activation_Softmax:
# Forward pass
    def forward(self, inputs):
# Remember input values
        self.inputs = inputs
# Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
        keepdims=True))
# Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
# Backward pass
    def backward(self, dvalues):
# Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
# Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, dvalues)):
# Flatten output array
            single_output = single_output.reshape(-1, 1)
# Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)

# Calculate sample-wise gradient
# and add it to the array of sample gradients
        self.dinputs[index] = np.dot(jacobian_matrix,
        single_dvalues)
# Common loss class
class Loss:
# Calculates the data and regularization losses
# given model output and ground truth values
    def calculate(self, output, y):
# Calculate sample losses
        sample_losses = self.forward(output, y)
# Calculate mean loss
        data_loss = np.mean(sample_losses)
# Return loss
        return data_loss
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
# Forward pass
    def forward(self, y_pred, y_true):
# Number of samples in a batch
        samples = len(y_pred)
# Clip data to prevent division by 0
# Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
# Probabilities for target values -
# only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
        ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
        )
# Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We’ll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        # Softmax classifier - combined Softmax activation
        # and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
# Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
# Forward pass
    def forward(self, inputs, y_true):
        # Output layer’s activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

        # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class something():
        pass

class Optimizer_SGD():
        # Initialize optimizer - set settings,
        # learning rate of 1. is default for this optimizer
        def __init__(self, learning_rate=1., decay=0.):
                self.learning_rate = learning_rate
                self.current_learning_rate = learning_rate
                self.decay = decay
                self.iterations = 0
        # Call once before any parameter updates
        def pre_update_params(self):
                if self.decay:
                        self.current_learning_rate = self.learning_rate *(1. / (1. + self.decay * self.iterations))
        # Update parameters
        def update_params(self, layer):
                layer.weights += -self.current_learning_rate * layer.dweights
                layer.biases += -self.current_learning_rate * layer.dbiases
                # Call once after any parameter updates
        def post_update_params(self):
                self.iterations += 1

class Optimizer_Adagrad:
# Initialize optimizer - set settings
        def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
                self.learning_rate = learning_rate
                self.current_learning_rate = learning_rate
                self.decay = decay
                self.iterations = 0
                self.epsilon = epsilon
# Call once before any parameter updates
        def pre_update_params(self):
                if self.decay:
                        self.current_learning_rate = self.learning_rate * \
                        (1. / (1. + self.decay * self.iterations))
        
        def update_params(self, layer):
# If layer does not contain cache arrays,
# create them filled with zeros
                if not hasattr(layer, 'weight_cache'):
                        layer.weight_cache = np.zeros_like(layer.weights)
                        layer.bias_cache = np.zeros_like(layer.biases)
                # Update cache with squared current gradients
                layer.weight_cache += layer.dweights**2
                layer.bias_cache += layer.dbiases**2
                # Vanilla SGD parameter update + normalization
                # with square rooted cache
                layer.weights += -self.current_learning_rate * \
                layer.dweights / \
                (np.sqrt(layer.weight_cache) + self.epsilon)
                layer.biases += -self.current_learning_rate * \
                layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)
                # Call once after any parameter updates
        def post_update_params(self):
                self.iterations += 1


class Optimizer_RMSprop():

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
        rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))

    # all weights are added to the cache, however bgger ones move the
    #  average to be smaller, smaller ones to be bigger 
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

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