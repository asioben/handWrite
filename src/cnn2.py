"Superior Model of Convolutional Neural Network 1.0"

#machine learning library
import torch as pt
from torch import nn
import numpy as np

#loading library
import load

#device = pt.accelerator.current_accelerator().type if pt.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

class SCNN(nn.Module):
    def __init__(self, channel, filter_size, mlp_size, conv_size, expected_input_size):
        super().__init__()
        self.convolution = []
        self.perceptron = []
        self.conv_size = conv_size
        self.perceptron_size = len(mlp_size)

        for i in range(conv_size):
          tmp_filter_size = filter_size * pow(2,(i+1))
          channel_size = filter_size * pow(2,(i+1))
          if i == 0:
             channel_size = channel
          convolution = nn.Sequential(
            nn.Conv2d(in_channels=channel_size,out_channels=tmp_filter_size*2,kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm2d(tmp_filter_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
           )
          self.convolution.append(convolution)

        for j in range(self.perceptron_size):
          if j == (self.perceptron_size - 1):
             perceptron = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(mlp_size[j - 1], mlp_size[j]),
                nn.Softmax(dim=1)
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


    def forward_propagation(self, x):
        for i in range(self.conv_size):
            x = self.convolution[i](x)
            print(np.shape(x))

        for j in range(self.perceptron_size):
           if j == (self.perceptron_size - 1):
             break
           
           x = self.perceptron[j](x)
           print(np.shape(x))
        
        logits = self.perceptron[-1](x)
        print(np.shape(logits))
        return logits
    

    #def back_propagation(self,y):
       
x = load.x_test[0]
#load.show_img([load.x_test[0]],"...")
x = pt.tensor(x,dtype=pt.float32)
x = pt.reshape(x,(1,1,28,28))
#print(x.shape)
model = SCNN(1,8,(128,64,10),2,3136)#.to(device)
print(model.forward_propagation(x))