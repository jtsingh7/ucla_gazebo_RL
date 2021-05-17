import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn

# INITIALIZE NETWORK WEIGHS
def init_weights(m):
    if isinstance(m, nn.Linear):
       nn.init.normal_(m.weight, mean=0, std=0.1)
       nn.init.constant_(m.bias, 0.1)


# PLOT FUNCTION
def plot_learning_curve(x, scores, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title(('Running average of previous 100 scores'))
    plt.savefig(filename)

