"Load the data "

#std lib
import struct 
from array import array
import random

#thrid-party lib
import matplotlib.pyplot as plt
import numpy as np


#Loader class
class Loader(object):
    def __init__(self,training_img_fp,training_label_fp,test_img_fp,test_label_fp):
        self.training_img_fp = training_img_fp
        self.training_label_fp = training_label_fp
        self.test_img_fp = test_img_fp
        self.test_label_fp = test_label_fp

#file reading functions
    def read_img(self,filepath):
        with open(filepath,'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Wrong magic number, expect 2051 but got {}".format(magic))
            img_data = array("B",file.read())
        images = []
        for i in range(size):
            images.append([0]*rows*cols)
        for i in range(size):
            img = np.array(img_data[i*rows*cols:(i+1)*rows*cols])
            img = img.reshape(784)
            images[i] = img.astype(float)
        return images
    
    def read_label(self,filepath):
        with open(filepath,'rb') as file:
            magic, size = struct.unpack('>II',file.read(8))
            if magic != 2049:
                raise ValueError("Wrong magic number, expect 2049 but got {}".format(magic))
            return array("B",file.read())
        
    def normalize(self,x):
        for pixels in x:
            pixels /= 255
            for i in range(len(pixels)):
                if pixels[i] > 0.0:
                   pixels[i] = 1.0
        return x
        
    def load_data(self):
        x_train = self.read_img(self.training_img_fp)
        x_train = self.normalize(x_train)
        y_train = self.read_label(self.training_label_fp)
        x_test = self.read_img(self.test_img_fp)
        x_test = self.normalize(x_test)
        y_test = self.read_label(self.test_label_fp)
        return (x_train,y_train),(x_test,y_test)
    
#outside the class
#some function to show the different images
def show_img(images,titles):    
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images,titles):
        img = x[0]
        img = np.reshape(img,(28,28))
        title = x[1]
        plt.subplot(rows,cols,index)
        plt.imshow(img)
        if(title != ''):
            plt.title(title,fontsize=15)
        index += 1
    plt.show()

dataloader = Loader("../data/train-images.idx3-ubyte","../data/train-labels.idx1-ubyte","../data/t10k-images.idx3-ubyte","../data/t10k-labels.idx1-ubyte")
(x_train,y_train),(x_test,y_test) = dataloader.load_data()

images_show = []
titles_show = []
for i in range(0,10):
    r = random.randint(1,60000)
    images_show.append(x_train[r])
    titles_show.append("Training:" + str(r) + "=> " + str(y_train[r]))

for i in range(0,5):
    r = random.randint(1,10000)
    if i == 0: r = 6020
    images_show.append(x_test[r])
    titles_show.append("Test:" + str(r) + "=> " + str(y_test[r]))

if __name__ == '__main__':
    show_img(images_show,titles_show)
    #print(x_test[6020])
    print(x_test[6020])