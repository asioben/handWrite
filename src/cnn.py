"Model of Convolutional Neural Network 1.0"

#I don't know what we'll be without this GOATesque library
import numpy as np

#dependancies
import nn
import load

#class
from nn import NeuralNetwork

class CNN(NeuralNetwork):
    def __init__(self,filters_size,sizes):
        super().__init__(sizes)
        self.filters_size = filters_size
        #the dimension of the filter is as such: (filters_size,3,3), it's a 3D array(tensor)
        #here the init
        self.filters = [np.random.randn(filters_size ,3,3) / 9 for i in range(2)]
        self.conv_biases = [np.zeros((filters_size * i,)) for i in range(1,3)]
        self.max_pool = []
        self.shapes = []
        self.conv_out = []
        self.region = []


    def convolution(self, input):
        h, w, d = input.shape # d for depth again !
        output = np.zeros((h - 2, w - 2, self.filters_size))
        regions = []
        for i in range(h - 2):
            for j in range(w - 2):
                if d > 1: 
                    region = input[i: (i + 3), j: (j + 3), :]               
                    output[i, j] = np.vdot(region.T,self.filters[1]) + self.conv_biases[1]
                else:
                    input = np.reshape(input,(28,28))
                    region = input[i: (i + 3), j: (j + 3)]
                    output[i, j] = np.sum(region * self.filters[0], axis = (1,2)) + self.conv_biases[0]
                regions.append(region)

        self.conv_out.append(output)
        self.region.append(regions)
                #output[i, j] = np.sum(region * self.filters, axis = (1,2))
                #output[i, j] = np.einsum('ijk,ijk',region,self.filters)

        return output
    
    def pool(self, input):
        h, w, d = input.shape # d for depth -\__:-:_/-
        h_ = h // 2
        w_ = w // 2

        output = np.zeros((h // 2, w // 2, d))
        max_pool = []

        for i in range(h_):
            for j in range(w_):
                region = input[(i * 2) : ((i * 2) + 2), (j * 2) : ((j * 2) + 2)]
                output[i , j] = np.amax(region, axis=(0,1))
                row, col, _ = np.where(output[i, j] == region)
                element = (row[0], col[0])
                max_pool.append(element)
        self.max_pool.append(max_pool)

        return output
    
    def convolutional_layer(self, input):
         #conv(8)
        output = self.convolution(input)
        self.shapes.append(output.shape)
        #first reLU
        output = nn.reLU(output)
        #first maxpool
        output = self.pool(output)
        self.shapes.append(output.shape)

        return output
    
    def perceptron(self, input):
        input = np.reshape(input,(self.filters_size * 5 * 5, 1))
        #print(input)
        #print(input.shape)
        output = super().forward_propagation(input)
        return output
    
    def forward_propagation(self, input):
        for i in range(2):
            output = self.convolutional_layer(input)
            if(i < 1):
                self.filters_size *= 2
            input = output
        output = self.perceptron(input)

    def back_pool(self, dO, step, shape): 
        #dI stands for dInputs

        dI = np.zeros(self.shapes[shape - 1]) 
        dO = np.reshape(dO,self.shapes[shape])
        #dO stands for dOutputs
        h, w, _ = dO.shape

        counter = 0
        for i in range(h):
            for j in range(w):
                dI[self.max_pool[step][counter]] = dO[i][j]
                counter += 1

        return dI

    #return in this order ->
    #dI, dW, dB
    def back_conv(self, dO, step, shape):
        dB = np.sum(dO,axis=(0,1))
        dW = np.zeros(self.filters[step].shape)
        dI = np.zeros(self.shapes[shape])
        h, w, d = dO.shape
        h_, w_, d_ = dI.shape
        #tmp = np.zeros((h,1))
        
        counter = 0
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    if i < (h_ - 2) and j < (w_ - 2) and k < d_ and step != 0:
                       dI[i:i+3, j:j+3, k] += dO[i,j,k] * np.rot90(self.filters[step][k],2)
                    dW += self.region[step][counter].T * dO[i,j,k]
                counter += 1

        return dI,dW,dB

    def update(self, dW, dB, step):
        self.filters[step] -= self.learning_rate * dW
        self.conv_biases[step] -= self.learning_rate * dB


    def back_propagation(self, y):
        #print(self.shapes)
        self.learning_rate = 0.01
        #backward
        dO = super().back_propagation(y)
        #SGD
        super().update()
        #print()
        shape = -1
        for i in reversed(range(2)):
            """if i == 1:
                for j in range(len(self.shapes)):
                    print(self.shapes[j])"""
            dO = self.back_pool(dO,i,shape)
            dO = back_reLU(dO, self.conv_out[i])
            dO, dW, dB = self.back_conv(dO,i,i)
            self.update(dW,dB,i)
            shape -= 2    


#miscellanous functions
def back_reLU(a, b):   
   #a represents the tensor that'd get transformed
   #b represents the mask, the sets of 0 and 1
   return a * (b > 0)
    
"""cnn = CNN(8,(400,64,10))
input = np.reshape(load.x_train[0],(28,28,1))
cnn.forward_propagation(input)
cnn.back_propagation(np.reshape(load.y_train[0],(1,1)))"""
"""img = load.x_train[0]
string = "test"
load.show_img([img,img,img,img,img],[string,string,string,string,string])"""