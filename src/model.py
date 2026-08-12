"Model of Neural Network 1.0"

#linking
import load
import nn
import cnn
import cnn2 

#std lib
import numpy as np
import json
import torch

#data to train and test the model
x_train = load.x_train
y_train = load.y_train
x_test = load.x_test
y_test = load.y_test

#functions for the panel of control
errorCode = "Invalid input !"

menu = [
        "Menu:",
        "0: Quit",
        "1: Model",
        "2: Train",
        "3: Predict",
        "4: Download"
]

models_menu = [
    "Choose your model:",
    "1: Multilayer Perceptron / Neural Network (NN)",
    "2: Convolutional Neural Network (CNN)",
    "3: Superior Convolutional Neural Network (SCNN)"
    "0: Go Back -\__:-:_/-"
]

models_name = [
    "Multilayer Perceptron / Neural Network (NN)",
    "Convolutional Neural Network (CNN)",
    "Superior Convolutional Neural Network (SCNN)",
    "Nothing :("
]

model_filepaths = [
    "nn_model.json",
    "cnn_model.json"
    "scnn_model_half_gray.pth"
]

def print_(to_Print):
    
    for text in to_Print:
        print(text)

def test_NAN(test,text,to_Print):
    while True:
        try:
            test = int(input(text))
            return test
        except ValueError:
            print(errorCode)
            print_(to_Print)

def check_file_and_write(filepath):
    print("I'll create a new file: " + filepath)
    with open(filepath, "w") as file:
        file.write("")

def number_expectation(to_Print,small,large,command):
    if command > large or command < small:
            print(errorCode)
            print_(to_Print)
            #command = test_NAN(command,"What's your command: ",to_Print)
            
    #return True

def main():
    size = (784,128,64,10)
    nn_model = nn.NeuralNetwork(size)
    cnn_model = cnn.CNN(8,(400,64,10))
    scnn_model = cnn2.SCNN(1,8,(128,64,10),2,3136)
    model = None
    menu_command = -1
    model_command = -1
    current_model = -1
    filepath = ""

    while True:
        print("We are using : ")
        current_model = model_command
        if current_model != -1:
            current_model = model_command - 1 
        print(models_name[current_model])
        if(menu_command != 1):
            print_(menu)
            menu_command = test_NAN(menu_command,"What's your command: ",menu)
            number_expectation(menu,0,4,menu_command)
        #print(command)
        if menu_command == 0:
            print("Bye...")
            break

        elif menu_command == 1:
            choose_model = True
            while choose_model == True:
                print_(models_menu)
                model_command = test_NAN(model_command,"Whats's your command: ",models_menu)
                number_expectation(models_menu,0,2,model_command)
                
                if model_command == 0:
                    print("Bye")
                    choose_model = False
                    menu_command = -1
                elif model_command == 1:
                    model = nn_model
                    print("The model in usage is the Multilayer Perceptron / Neural Network (NN) ")
                    choose_model = False
                    menu_command = -1
                elif model_command == 2:
                    model = cnn_model
                    print("The model in usage is the Multilayer Perceptron + Convolutional Neural Network (CNN) ")
                    choose_model = False
                    menu_command = -1
                elif model_command == 3:
                    model = scnn_model
                    print("The model in usage is the Superior Multilayer Perceptron + Convolutional Neural Network (SCNN) ")
                    choose_model = False
                    menu_command = -1
                
        elif menu_command == 2:
            if(model == None):
                print("There is no model to train")
                print("Go ahead choose one")
                menu_command = 1
            elif model_command == 3:
                print("Model in training...")
                model.train_(x_train,y_train,0.01,60,16)
            else:
               print("Model in training...")
               model.train(x_train,y_train,0.01,60,16)

        elif menu_command == 3:
            if(model == None):
                print("We don't have a model to make prediction with")
                print("Please, consider choosing one before !")
                menu_command = 1
            else:
               print("We gonna predict...")
               if(model.trained == False):
                  if(model_command == 3):
                      model = torch.load("scnn_model.pth",weights_only=False)
                      model.eval()
                  else:
                    model = load_network(model_filepaths[current_model],(current_model + 1))
               model.predict(x_test,y_test)

        elif menu_command == 4:
            if(model == None):
                print("What model do you even want to download")
                print("Choose one before doing that")
                menu_command = 1
            else:
                filepath = model_filepaths[current_model]
                check_file_and_write(filepath)
                if(model_command == 3):
                    torch.save(model,filepath)
                else:
                   model.download_network(filepath)

#load a neural network

def load_network(filepath,model_type):
    network = None
    
    with open(filepath,"r") as file:
        data = json.load(file)

    if(model_type == 1):
      network = nn.NeuralNetwork(data["sizes"])
      network.weights = [np.array(w) for w in data["weights"]]
      network.biases = [np.array(b) for b in data["biases"]]

    elif (model_type == 2):
        network = cnn.CNN(data["filters_size"],data["sizes"])
        network.filters = [np.array(f) for f in data["filters"]]
        network.conv_biases = [np.array(c) for c in data["conv_biases"]]
        network.weights = [np.array(w) for w in data["weights"]]
        network.biases = [np.array(b) for b in data["biases"]]

    return network

if __name__ == "__main__":
    main()

    #################
    """
        ############
    I DONT NEED CLASSES !!!!
    I NEED NAMESPACE :( !!!!!
        #############
    """

    ##################