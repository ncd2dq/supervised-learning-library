'''This library introduces a simple Neural Network (NN) object that can be
   any size/depth for supervised learning.
'''

import numpy as np

class NN(object):
    def __init__(self,size=None):
        '''Expects size to be a tuple'''
        if size == None:
            self.shape = (3,1)
            self.layerCount = len(self.shape) - 1
        else:
            self.shape = size
            self.layerCount = len(self.shape) - 1
            
        self.weights = []
        for l0,l1 in zip(self.shape[:-1],self.shape[1:]):
            new_weight_matrix = 2 * np.random.random((l0 + 1,l1)) - 1
            self.weights.append(new_weight_matrix)
            
        self.activation_dictionary = self.create_activation_dict()
        self.activation = 'sig'
        self.activation_function = self.activation_dictionary[self.activation]

        self._layerInput  = []
        self._layerOutput = []

        self._layerDelta = []
        self._layerError = []

    def create_activation_dict(self):
        '''Creates a dictionary of all possible activation functions'''
        def sig(z,deriv=False):
            output = 1 / (1 + np.exp(-z))
            if deriv:
                return output * (1 - output)
            else:
                return output
               
        def ReLU(z,deriv=False):
            output = z
            if deriv:
                return 1
            else:
                return output
        func_dict = {'sig':sig,'ReLU':ReLU}
        return func_dict

    def set_activation(self,activation):
        '''set the activation function'''
        self.activation = activation
        self.activation_function = self.activation_dictionary[self.activation]

    def create_bias(self,inpt):
        '''appends a col vector of 1s to input data'''
        inpt_rows = len(inpt)
        bias = np.ones((inpt_rows,1))
        inpt_with_bias = np.hstack((inpt,bias))
        return inpt_with_bias
    
    def forward_pass(self,inpt):
        '''Runs the data through the network through dot products'''

        self._layerInput =  []
        self._layerOutput = []

        for index,syn in enumerate(self.weights):
            if index == 0:
                inpt_with_bias = self.create_bias(inpt)
                self._layerInput.append(inpt_with_bias)
                output = np.dot(inpt_with_bias,syn)
                output = self.activation_function(output)
            else:
                inp = self._layerOutput[-1]
                inpt_with_bias = self.create_bias(inp)
                output = np.dot(inpt_with_bias,syn)
                output = self.activation_function(output)

            self._layerOutput.append(output)

    def backwards_prop(self,output,LR):
        '''Computes the back propogation step and updates the weights'''
        self._layerDelta = []
        self._layerError = []

        for index in reversed(range(self.layerCount)):
            if index == self.layerCount-1:
                error = output - self._layerOutput[-1]
                delta = error * self.activation_function(self._layerOutput[index],deriv=True)

            else:
                delta_upstream = self._layerDelta[-1]
                syn_upstream = self.weights[index+1]
                error = np.dot(delta_upstream,syn_upstream.T)
                error = error[:,:-1]
                delta = error * self.activation_function(self._layerOutput[index]) * LR
             
            self._layerError.append(error)
            self._layerDelta.append(delta)

        for index in reversed(range(self.layerCount)):
            if index == 0:
                inpt = self._layerInput[index]
                update = np.dot(inpt.T,self._layerDelta[len(self._layerDelta) - 1 - index])
                self.weights[index] += update * LR
            else:
                inpt = self._layerOutput[len(self._layerOutput) - 1 - index]
                inpt_with_bias = self.create_bias(inpt)
                update=  np.dot(inpt_with_bias.T,self._layerDelta[len(self._layerDelta) - 1 - index])
                self.weights[index] += update * LR
            
    def train(self,inpt,output,steps=10000,LR=0.01,activation = 'sig'):
        self.set_activation(activation)
        for i in range(steps):
            self.forward_pass(inpt)
            self.backwards_prop(output,LR)
            if i%(steps / 10) == 0:
                print('Current error: {}'.format(str(np.sum(self._layerError[-1]))))
        print('\nComplete, Final Result: {}'.format(str(self._layerOutput[-1])))


if __name__ == '__main__':
    n = NN((3,4,4,4,4,1))
    testingData = np.array([[1,0,1],[1,0,0]])
    y = np.array([[1],[0]])
    n.train(testingData,y,steps=1000000,LR=0.0001,activation = 'ReLU')
