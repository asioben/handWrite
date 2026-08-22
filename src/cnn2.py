"Superior Model of Convolutional Neural Network 1.0"

#machine learning library

#torch
import torch as pt
from torch import nn
from torch import optim

#numpy
import numpy as np

#loading library
import load

import nn as util

import matplotlib.pyplot as plt

#device = pt.accelerator.current_accelerator().type if pt.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

class SCNN(nn.Module):
    def __init__(self, channel, filter_size, mlp_size, conv_size, expected_input_size):
        super().__init__()
        self.convolution = nn.ModuleList([])
        self.perceptron = nn.ModuleList([])
        self.conv_size = conv_size
        self.perceptron_size = len(mlp_size)

        for i in range(conv_size):
          tmp_filter_size = filter_size * pow(2,(i+1))
          channel_size = filter_size * pow(2,(i+1))
          if i == 0:
             channel_size = channel
          convolution = nn.Sequential(
            nn.Conv2d(in_channels=channel_size,out_channels=tmp_filter_size*2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(tmp_filter_size*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
           )
          self.convolution.append(convolution)

        for j in range(self.perceptron_size):
          if j == (self.perceptron_size - 1):
             perceptron = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(mlp_size[j - 1], mlp_size[j]),
                #nn.Softmax(dim=1),
             )
          
          elif j == 0:
             perceptron = nn.Sequential(
                nn.Flatten(),
                nn.Linear(expected_input_size, mlp_size[0]),
                nn.ReLU(),
           )

          else:
           perceptron = nn.Sequential(
            nn.Linear(mlp_size[j - 1], mlp_size[j]),
            nn.ReLU(),
            #nn.Dropout(0.5)
          )

          self.perceptron.append(perceptron)

          #nn.ModuleList(self.convolution)
          #nn.ModuleList(self.perceptron)
          for param in self.parameters():
            param.requires_grad = True

        self.trained = False


    def forward_propagation(self, x):
        for conv in self.convolution:
            x = conv(x)
            #print(np.shape(x))

        for j in range(self.perceptron_size):
           if j == (self.perceptron_size - 1):
             break
           
           x = self.perceptron[j](x)
           #print(np.shape(x))
        
        logits = self.perceptron[-1](x)
        #print(np.shape(logits))
        return logits
    

    def back_propagation(self, output, y):
       loss_fn = nn.CrossEntropyLoss()
       loss = loss_fn(output,y)
       optimizer = optim.SGD(self.parameters(),lr=self.learning_rate)
       #print(self.parameters())

       #loss.requires_grad = True
       optimizer.zero_grad()
       loss.backward()

       optimizer.step()
       self.loss += loss.item()
       #optimizer.zero_grad()

       #print(list(self.convolution[0][0].parameters())[-1].grad)

    def train_(self,X,y,learning_rate,batch_size,epoch):
        self.X = X #input
        self.y = y #output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.trained = True
        self.loss = 0
        epochs_text = []
        for epoch in range(self.epoch):
            permutation = pt.randperm(len(self.X))
            self.X = util.shuffle_dataset(self.X,permutation)
            self.y = util.shuffle_dataset(self.y,permutation)
            y_true = util.transform_labels(self.y)
            number_batches = len(self.X) // self.batch_size
            losses = []
            for j in range(number_batches):
                #print(epoch,j)
                key = j * self.batch_size #where to chop the batch
                X_batch = pt.tensor(self.X[key:key+self.batch_size]).float()
                X_batch = pt.reshape(X_batch,[60,1,28,28])
                #X_batch = pt.transpose(X_batch,-1,0)
                y_batch = pt.tensor(y_true[key:key+self.batch_size])
                y_batch_true = pt.tensor(self.y[key:key+self.batch_size])
                y_batch_true = pt.transpose(y_batch_true,-1,0)
                y = self.forward_propagation(X_batch)
                self.back_propagation(y,y_batch)
                if((j % 200) == 0):
                    text = "Train step (" + str(j) + "): " + " Loss: " + str(self.loss / 100)
                    print(text)
                    losses.append((self.loss/100))
                    self.loss = 0
            text = "Epoch (" + str(epoch) + "): " + " Loss: " + str(losses[-1] / 100)
            self.loss = 0
            epochs_text.append(text)
            print(text)
            print("//////////////////////////////////")
        for z in range(len(epochs_text)):
            print(epochs_text[z])
        plt.plot(losses, label="Training Loss")
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show() 

    def predict(self,X,y):
        X = pt.tensor(X).float()
        y = pt.tensor(y).float()
        indice = pt.randint(0,len(X),(1,1)) #image to recognize
        img = pt.reshape(X[indice],(1,1,28,28))
        output = self.forward_propagation(img)
        digit = pt.argmax(output)
        print("Prediction: " + str(digit))
        print("Actual: "+ str(y[indice]))
        title = str(indice) + " : " + str(y[indice])
        load.show_img([img,img,img,img,img],[title,title,title,title,title])     

    def test(self,X,y,size):
       success = 0
       for i in range(size):
         x = X[i]
         x = pt.tensor(x,dtype=pt.float32)
         x = pt.reshape(x,(1,1,28,28)) 
         output = pt.argmax(self.forward_propagation(x))
         if(output == pt.tensor(y[i])):
           success += 1

       print((success/100))
       
def main():
    model = pt.load("../network/scnn_model_2.pth",weights_only=False)
    model.eval()
    model.test(load.x_test,load.y_test)
    #model.predict(load.x_test,load.y_test)

if __name__ == "__main__":
   main()

"""
I train two model with 
two types of the same data:

Grayscale
Binary (Black and white data)


The binary model: scnn_model.pth

Have 98.21% accuracy with binary data
Have 97.62% accuracy with grayscale data


The grayscale model: scnn_model_2.pth

Have 96.57% accuracy with binary data
Have 98.61% accuracy with grayscale data
"""
