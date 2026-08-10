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
       optimizer = optim.SGD(self.parameters(),lr=0.01)
       #print(self.parameters())

       #loss.requires_grad = True
       optimizer.zero_grad()
       loss.backward()

       optimizer.step()
       #optimizer.zero_grad()

       print(list(self.convolution[0][0].parameters())[-1].grad)

       
       
x = load.x_test[0]
#load.show_img([load.x_test[0]],"...")
x = pt.tensor(x,dtype=pt.float32)
x = pt.reshape(x,(1,1,28,28))
#print(x.shape)
model = SCNN(1,8,(128,64,10),2,3136)#.to(device)
#output = pt.argmax(model.forward_propagation(x))
output = model.forward_propagation(x)
print(output)
#print(len(list(model.parameters())))
#for name, param in model.named_parameters():
   #print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
model.back_propagation(output,pt.tensor([load.y_test[0]],dtype=pt.long))
