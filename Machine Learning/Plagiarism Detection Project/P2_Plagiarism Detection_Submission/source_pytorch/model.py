# torch imports
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        
        self.input_features = input_features #3
        self.hidden_dim = hidden_dim # 50-100
        self.output_dim = output_dim #1
        
                     
        # defining 2 linear layers
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU(1)
        
        self.fc1 = nn.Linear(self.input_features, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 100)
        self.out = nn.Linear(100, self.output_dim)
        
        #dropout
        self.drop = nn.Dropout(0.2)
        
        # sigmoid layer
        self.sig = nn.Sigmoid()
        
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        out = self.relu(self.fc1(x)) # activation on hidden layer
        out = self.drop(out)
        out = self.fc2(out)
        out = self.prelu(out)
        out = self.out(out)
        return self.sig(out) # returning class score
    
    