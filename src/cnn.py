"Model of Convolutional Neural Network 1.0"

#I don't know what we'll be without this GOATesque library
import numpy as np

#dependancies
import nn
import load

#third lib
import json

#class
from nn import NeuralNetwork

class CNN(NeuralNetwork):
    def __init__(self,filters_size,sizes):
        super().__init__(sizes)
        self.og_filters_sizes = filters_size
        self.filters_size = filters_size
        #the dimension of the filter is as such: (filters_size,3,3), it's a 3D array(tensor)
        #here the init
        self.filters = [np.random.randn(filters_size * i ,3,3,np.pow(filters_size,i-1)) / 9 for i in range(1,3)]
        self.conv_biases = [np.zeros((filters_size * i,)) for i in range(1,3)]
        self.max_pool = [0,0]
        self.shapes = []
        self.conv_out = []
        self.cnn = True
        self.input = None


    def convolution(self, input):
        if(input.ndim == 2):
           shape = input.shape
           size = int(np.sqrt(shape[0]))
           input = np.reshape(input,(shape[1],size,size,1))
           #print(input.shape)
        b, h, w, d = input.shape # d for depth again !
        output = np.zeros((b,h - 2, w - 2, self.filters_size))
        step = 0
        for k in range(b):
         for i in range(h - 2):
            for j in range(w - 2):
              for f in range(self.filters_size):
                if d > 1:
                    region = input[k, i: (i + 3), j: (j + 3), :]
                    #region_ = region.flatten()
                    step = 1
                else:
                    region = input[k, i: (i + 3), j: (j + 3), 0]
                    #region_ = np.reshape(region,(3,3,1))
                    step = 0
                    self.input = input
                    self.filters[step] = self.filters[step].squeeze()
                #filters_ = self.filters[step].flatten()
                #print(region_.shape,filters_.shape)
                #print(output[k,i,j,f].shape,region.shape,self.filters[step][f].shape,self.conv_biases[step][f].shape)
                output[k, i, j, f] = np.sum(region * self.filters[step][f]) + self.conv_biases[step][f]

        if(self.trained == True):   
           self.conv_out.append(output)

        return output
    
    def pool(self, input, step):
        b, h, w, d = input.shape # d for depth -\__:-:_/-
        h_ = h // 2
        w_ = w // 2

        output = np.zeros((b, h_, w_, d))
        self.max_pool[step] = np.empty((b,h_,w_,d,2), dtype=int)

        for k in range(b):
          for i in range(h_):
            for j in range(w_):
                region = input[k, (i * 2) : ((i * 2) + 2), (j * 2) : ((j * 2) + 2)]
                output[k, i , j, :] = np.max(region, axis=(0,1))
                if(self.trained == True):
                   for c in range(d):
                    index = np.argmax(region[:,:,c])
                    row = index // 2
                    col = index % 2
                    self.max_pool[step][k,i,j,c] = (row,col)

        return output
    
    def convolutional_layer(self, input, step):
         #conv(8)
        output = self.convolution(input)
        self.shapes.append(output.shape)
        #first reLU
        output = nn.reLU(output)
        #first maxpool
        output = self.pool(output,step)
        self.shapes.append(output.shape)

        return output
    
    def perceptron(self, input):
        shape = self.sizes[0]
        input = np.reshape(input,(shape, input.shape[0]))
        output = super().forward_propagation(input)
        return output
    
    def forward_propagation(self, input):
        #print(input.shape)
        for i in range(2):
            output = self.convolutional_layer(input,i)
            if(i < 1):
                self.filters_size *= 2
            input = output
        output = self.perceptron(input)
        self.filters_size = self.og_filters_sizes

        return output

    def back_pool(self, dO, step, shape): 
        #dI stands for dInputs

        dI = np.zeros(self.shapes[shape - 1]) 
        dO = np.reshape(dO,self.shapes[shape])
        #dO stands for dOutputs
        b, h, w, d = dO.shape
        #btw b always stand for batch size
        
        for k in range(b):
         for i in range(h):
            for j in range(w):
             for c in range(d):
                mi, mj = self.max_pool[step][k,i,j,c]
                dI[k,i*2+mi,j*2+mj,c] = dO[k,i,j,c]

        return dI

    #return in this order ->
    #dI, dW, dB
    def back_conv(self, dO, step, shape):
        dB = np.zeros(self.conv_biases[step].shape)
        dW = np.zeros(self.filters[step].shape)
        dI = np.zeros(self.shapes[shape])
        b, h, w, d = dO.shape
        b, h_, w_, d_ = dI.shape


        
        for l in range(b):
         
         counter = 0
         dB += np.sum(dO[l],axis=(0,1))

         for i in range(h):
            for j in range(w):

                for k in range(d):
        
                   if (i+3) <= h_ and (j+3) <= w_:

                        if step != 0:
                          
                          for c in range(d_):
                            dI[l, i:i+3, j:j+3, c] += dO[l, i,j,k] * np.rot90(self.filters[step][k,:,:,c],2)
                    
                        else:
                            dI[l, i:i+3, j:j+3, 0] += dO[l, i,j,k] * np.rot90(self.filters[step][k],2)

                   region = self.input[l, i:i+3, j:j+3, :].squeeze()
                   if k < dW.shape[0]:
                      if step == 0:
                        dW[k] += region * dO[l,i,j,k]
                      else:
                         for e in range(dW.shape[-1]):
                            dW[k,:,:,e] += region * dO[l,i,j,k]
                      
                counter += 1

        dW /= b
        dB /= b

        """print(np.linalg.norm(dW))
        print(np.linalg.norm(dI))"""

        """print(np.isnan(dO).any())
        print(np.max(np.abs(dO)))"""
        

        return dI,dW,dB

    def update(self, dW, dB, step):
        self.filters[step] -= self.learning_rate * dW
        self.conv_biases[step] -= self.learning_rate * dB


    def back_propagation(self, y):
        #print(self.shapes)
        #self.trained = True
        #self.learning_rate = 0.01
        shape = y.shape
        y = np.reshape(y,(shape[1],shape[0],1))
        #print(y.shape)

        for k in range(y.shape[0]):
          #backward
          dO, loss = super().back_propagation(y[k])
          #SGD
          super().update()
          #print(dO.shape)

        #print()
        shape = -1
        for i in reversed(range(2)):
            if i == 1:
                """for j in range(len(self.shapes)):
                    print(self.shapes[j])"""
            dO = self.back_pool(dO,i,shape)
            #print(dO.shape)
            dO = back_reLU(dO, self.conv_out[i])
            dO, dW, dB = self.back_conv(dO,i,i)
            #print(dO.shape)
            self.update(dW,dB,i)
            shape -= 2  

        #self.filters_size = self.og_filters_sizes 
        self.max_pool = [0,0]
        self.shapes = []
        self.conv_out = []
        self.input = None

        return loss
    
    def download_network(self, filepath):
       data = {
            "sizes": self.sizes,
            "filters_size": self.og_filters_sizes,
            "filters": [f.tolist() for f in self.filters],
            "conv_biases": [c.tolist() for c in self.conv_biases],
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
       with open(filepath,"w") as file:
            json.dump(data,file)
       

#miscellanous functions
def back_reLU(a, b):   
   #a represents the tensor that'd get transformed
   #b represents the mask, the sets of 0 and 1
   return a * (b > 0)
    
"""cnn = CNN(8,(400,64,10))
cnn.predict(load.x_test,load.y_test)"""
"""img = load.x_train[0]
string = "test"
load.show_img([img,img,img,img,img],[string,string,string,string,string])"""


"""
WHAT I'VE DONE SO FAR:
-CLEANUP CONV OUTPUT CALCULATION
-ADD BATCH SIZE DIMENSION
for the moment every shape seem correct 
"""
"""
WHAT WE GONNA DO
-TEST AND TRAIN !!!!!!!!
"""