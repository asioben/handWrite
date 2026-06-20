"Model of Convolutional Neural Network 1.0"

#I don't know what we'll be without this GOATesque library
import numpy as np

#dependancies
import nn
import load

class CNN():
    def __init__(self,filters_size,sizes):
        #super().__init__(sizes)
        self.filters_size = filters_size
        #the dimension of the filter is as such: (filters_size,3,3), it's a 3D array(tensor)
        #here the init
        self.filters = [np.random.randn(filters_size ,3,3) / 9 for i in range(2)]
        print(self.filters)
        print(len(self.filters))
        #self.weights = [np.random.randn(y,x) * np.sqrt(2/x) for x,y in zip(sizes[:-1], sizes[1:])]
        #self.biases = [np.zeros((y,1)) for y in sizes[1:]]


    def convolution(self, input):
        h, w, d = input.shape # d for depth again !
        output = np.zeros((h - 2, w - 2, self.filters_size))
        print(input.shape)
        for i in range(h - 2):
            for j in range(w - 2):
                if d > 1: 
                    region = input[i: (i + 3), j: (j + 3), :]               
                    output[i, j] = np.vdot(region.T,self.filters[1])
                else:
                    input = np.reshape(input,(28,28))
                    region = input[i: (i + 3), j: (j + 3)]
                    output[i, j] = np.sum(region * self.filters[0], axis = (1,2))
                #output[i, j] = np.sum(region * self.filters, axis = (1,2))
                #output[i, j] = np.einsum('ijk,ijk',region,self.filters)

        return output
    
    def pool(self, input):
        h, w, d = input.shape # d for depth -\__:-:_/-
        h_ = h // 2
        w_ = w // 2

        output = np.zeros((h // 2, w // 2, d))

        for i in range(h_):
            for j in range(w_):
                region = input[(i * 2) : ((i * 2) + 2), (j * 2) : ((j * 2) + 2)]
                output[i , j] = np.amax(region, axis=(0,1))

        return output
    
    def convolutional_layer(self, input):
         #conv(8)
        output = self.convolution(input)
        #first reLU
        output = nn.reLU(output)
        #first maxpool
        output = self.pool(output)
        print(output.shape)

        return output
    
    #def perceptron():
    
    def forward(self, input):
        for i in range(2):
            output = self.convolutional_layer(input)
            self.filters_size *= 2
            input = output
        
        

    
cnn = CNN(8,(400,64,10))
input = np.reshape(load.x_train[0],(28,28,1))
"""output = cnn.convolution()
output = cnn.pool(output)
print(output.shape)"""
cnn.forward(input)