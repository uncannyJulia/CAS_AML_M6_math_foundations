import numpy as np

# for regression is rms: ypred-y/sum(ypred)
# for classification is cross entropy loss: -log(conf(pred))
# its called cross entropy because we compare the softmaß output with the desired output distribution
# we log, because we want to reverse the exponentiation we used in softmax
# actually it just comes down to negative log of the outputconf that shaould have been the correct class


# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]
def simple_loss(input:list, labels:list):
    loss=np.sum(-(np.log(input)*labels))
    print(loss)

simple_loss(softmax_output, target_output)   # this one works for samples, not so much for batches 

softmax_outputs =   np.array([[0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]])
class_targets =     np.array([0, 1, 1])


def np_loss(inputs, labels):
    if len(labels.shape)==1:
        correct_confidences=inputs[
            range(len(inputs)), labels]


    elif len(labels.shape)==2:
        correct_confidences =np.sum(
            inputs*labels,
            axis=1)

    neg_log=-np.log(correct_confidences)
    avg_loss=np.mean(neg_log)
    print(avg_loss)

np_loss(softmax_outputs, class_targets)


import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]])

np_loss(softmax_outputs, class_targets)  

# soon we will encounter a zero dividion error because the log of 0 is undefined (when the cofidence is 0), 
# thus in the actuall class we can clip values to slightly above zero

class loss:
    def calculate(self, output, y):
        sample_losses=self.forward(output, y)
        data_loss=np.mean(sample_losses)
        return (data_loss)

class crossentropy_loss(loss):
    def forward(self, y_pred, y_true):
        samples=len(y_pred)
        y_pred_clip=np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clip[
            range(len(samples)), y_true]


        elif len(y_pred.shape)==2:
            correct_confidences =np.sum(
            y_pred_clip*y_true,
            axis=1)
        negative_log_likelihood=-np.log(correct_confidences)
        return negative_log_likelihood


loss_function = crossentropy_loss()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(f"cross_entr_class: {loss}")








