"Model of Neural Network 1.0"

#thrid-party lib
import numpy as np

#linking
import load

#std lib
import json 

class NeuralNetwork:
    def __init__(self,sizes):
        self.sizes = sizes
        #self.layer = len(self.sizes)
        self.biases = [np.zeros((y,1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) * np.sqrt(2/x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.trained = False
        self.cnn = False
        #result = 0
        """for i in range(len(self.weights)):
            result += self.weights[i].size
        #print(result)"""

    #this function is the activation of a function, when it "fires"
    #btw a stands for activation, w for weight and b for bias  
    def forward_propagation(self,X):
     self.z = []
     a = X
     self.a = [a]
     for i in range(len(self.weights)):
         z = np.dot(self.weights[i], a) + self.biases[i]
         self.z.append(z)
         if i == len(self.weights) - 1:
             a = softmax(z) #means we're at the output layer
         else:
             a = reLU(z) #means we're at the hidden layers
         self.a.append(a)
     return a
    
    #the function that computes the error and make the network learn
    def back_propagation(self,y):
         #dB reprendents the gradient of the biases and dW the gradients of the weights
         #don't forget dZ, we can say it represents the error
         #and also y are labels
         m = y.shape[1]
         loss = - np.log(self.a[-1][np.argmax(y)]) + 1e-9
         self.dW = [np.zeros_like(w) for w in self.weights]
         self.dB = [np.zeros_like(b) for b in self.biases]
         dZ = self.a[-1] - y
         for i in reversed(range(len(self.weights))):
             self.dW[i] = (1/m) * np.dot(dZ,self.a[i].T)
             self.dB[i] = (1/m) * np.sum(dZ,axis=1,keepdims=True)
             if i > 0:
                 dZ = np.dot(self.weights[i].T,dZ) * (self.z[i-1] > 0)
                 
         dZ = np.dot(self.weights[0].T,dZ)
         return dZ, loss
    
    #update the weights and biases depnding on the error
    def update(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.dB[i]

    def train(self,X,y,learning_rate,batch_size,epoch):
        self.X = X #input
        self.y = y #output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.trained = True
        epochs_text = []
        for epoch in range(self.epoch):
            permutation = np.random.permutation(len(self.X))
            self.X = shuffle_dataset(self.X,permutation)
            self.y = shuffle_dataset(self.y,permutation)
            y_true = transform_labels(self.y)
            number_batches = len(self.X) // self.batch_size
            model_accuracy = []
            loss = 0
            for j in range(100):
                print(epoch,j)
                key = j * self.batch_size #where to chop the batch
                X_batch = self.X[key:key+self.batch_size]
                X_batch = np.transpose(X_batch)
                y_batch = y_true[key:key+self.batch_size]
                y_batch = np.transpose(y_batch)
                y_batch_true = self.y[key:key+self.batch_size]
                y_batch_true = np.transpose(y_batch_true)
                y = self.forward_propagation(X_batch)
                loss += self.back_propagation(y_batch)
                if(self.cnn == False):
                    self.update()
                model_accuracy.append(accuracy(y,y_batch_true))
                if((j % 100) == 0):
                    network_accuracy = np.mean(model_accuracy) * 100
                    text = "Train step (" + str(j) + "): " + str(network_accuracy) + ("%")
                    print(loss / 100)
                    print(text)
                    loss = 0
            network_accuracy = np.mean(model_accuracy) * 100
            text = "Epoch (" + str(epoch) + "): " + str(network_accuracy) + ("%")
            epochs_text.append(text)
            print(text)
            print("//////////////////////////////////")
        for z in range(len(epochs_text)):
            print(epochs_text[z])
    
    def predict(self,X,y):
        indice = np.random.randint(len(X)) #image to recognize
        img = np.reshape(X[indice],(784,1))
        output = self.forward_propagation(img)
        digit = np.argmax(output)
        print("Prediction: " + str(digit))
        print("Actual: "+ str(y[indice]))
        title = str(indice) + " : " + str(y[indice])
        load.show_img([img,img,img,img,img],[title,title,title,title,title])

    def download_network(self,filepath):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filepath,"w") as file:
            json.dump(data,file)

#some miscellaneous function to compute
#its just some mathematical function
def reLU(z):
    return np.maximum(z,0)

def softmax(z):
     z = z - np.max(z,axis=0,keepdims=True)
     return np.exp(z) / np.sum(np.exp(z),axis=0,keepdims=True)

#these function are miscellanous too, but you will notice they're a bit different they are more like routine
def shuffle_dataset(dataset,indices):
    copy = dataset
    for i in range(len(dataset)):
        dataset[i] = copy[indices[i]]
    return dataset

def accuracy(output,label):
    output = output.T
    i = 0 
    success = 0
    for y in output:
        digit = np.argmax(y) #this function helps get the digit by finding the indices which contains the number with the highest probability
        if digit == label[i]:
            success += 1
        i += 1
    return success/output.shape[0]

def transform_labels(labels):
    true = []
    for label in labels:
        array = np.zeros(10)
        array[label] = 1.0
        true.append(array)
    return true