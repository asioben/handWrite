"Model of Convolutional Neural Network 1.0"

#I don't know what we'll be without this GOATesque library
import numpy as np

#dependancies
import nn

import load

class CNN():
    def __init__(self,filters_size):
        self.filters_size = filters_size
        #the dimension of the filter is as such: (filters_size,3,3), it's a 3D array(tensor)
        #here the init
        self.filters = np.random.rand(filters_size,3,3) / 9

    def convolution(self, input):
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.filters_size))

        for i in range(h -2):
            for j in range(w - 2):
                region = input[i: (i + 3), j:(j + 3)]
                output[i, j] = np.sum(region * self.filters, axis = (1,2))

        return output
    
cnn = CNN(8)
output = cnn.convolution(np.reshape(load.x_train[0],(28,28)))
print(output.shape)
