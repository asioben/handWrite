#import json
import numpy as np
from main import Data

def accuracy():
    digits = Data("../data/test_data.json",None)
    data = digits.load()

    data_ = []
    data_.append(data["NN"])
    data_.append(data["CNN"])
    data_.append(data["SCNN1"])
    data_.append(data["SCNN2"])

    data_ = np.reshape(data_,(4,100))

    success = [0,0,0,0]

    for i in range(4):
        for j in range(100):
            if((j % 10) == data_[i][j]):
                success[i] += 1

    print("Accuracy: ") 
    print("NN: " + str(success[0]) + "% !")
    print("CNN: " + str(success[1]) + "% !")
    print("SCNN1: " + str(success[2]) + "% !")
    print("SCNN2: " + str(success[3]) + "% !")


if __name__ == "__main__":
    accuracy()
