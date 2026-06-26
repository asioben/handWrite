"Model of Neural Network 1.0"

#linking
import load
import nn
import cnn

#std lib
import os 

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

models_ = [
    "Choose your model:",
    "1: Multilayer Perceptron / Neural Network (NN)",
    "2: Convolutional Neural Network (CNN)",
    "0: Go Back -\__:-:_/-"
]

models = [
    "Multilayer Perceptron / Neural Network (NN)",
    "Convolutional Neural Network (CNN)",
    "Nothing :("
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
    print("I'll create a new file, model.json")
    with open(filepath, "w") as file:
        file.write("")

def number_expectation(to_Print,small,large,command):
    while command > large or command < small:
            print(errorCode)
            print_(to_Print)
            #command = test_NAN(command,"What's your command: ",to_Print)
            
    #return True

def main():
    size = (784,128,64,32,10)
    nn_model = nn.NeuralNetwork(size)
    cnn_model = cnn.CNN(8,(size))
    model = None
    command = -1
    command_ = -1
    _command_ = -1
    filepath = ""
    trained = False

    while True:
        print("We are using : ")
        _command_ = command_
        if _command_ != -1:
            _command_ = command_ - 1 
        print(models[_command_])
        if(command != 1):
            print_(menu)
            command = test_NAN(command,"What's your command: ",menu)
            number_expectation(menu,0,4,command)
        #print(command)
        if command == 0:
            print("Bye...")
            break

        elif command == 1:
            choose_model = True
            while choose_model == True:
                print_(models_)
                command_ = test_NAN(command_,"Whats's your command: ",models_)
                number_expectation(models_,0,2,command_)
                
                if command_ == 0:
                    print("Bye")
                    choose_model = False
                    command = -1
                elif command_ == 1:
                    model = nn_model
                    print("The model in usage is the Multilayer Perceptron / Neural Network (NN) ")
                    choose_model = False
                    command = -1
                elif command_ == 2:
                    model = cnn_model
                    print("The model in usage is the Multilayer Convolutional Neural Network (CNN) ")
                    choose_model = False
                    command = -1
                
        elif command == 2:
            if(model == None):
                print("There is no model to train")
                print("Go ahead choose one")
                command = 1
            else:
               print("Model in training...")
               trained = True
               model.train(x_train,y_train,0.01,60,16)

        elif command == 3:
            if(model == None):
                print("We don't have a model to make prediction with")
                print("Please, consider choosing one before !")
                command = 1
            else:
               print("We gonna predict...")
               if(trained == False):
                  model = nn.load_network("model.json")
                  trained = True
               model.predict(x_test,y_test)

        elif command == 4:
            print("Closed command")
            break
            if(model == None):
                print("What model do you even whant to download")
                print("Choose one before doing that")
                command = 1
            else:
                filepath_or = input("Choose a filepath (tap y) or not: ")
                if filepath_or == "y":
                   filepath = input("Write the filepath: ")
                   while os.path.exists(filepath) == False:
                     if filepath != "y":
                       filepath = input("Be sure that file exist (tap y), write the filepath: ")
                     else: 
                        filepath = "model.json"
                        check_file_and_write(filepath)
                        break
                else:
                  filepath = "model.json"
                  check_file_and_write(filepath)
                model.download_network(filepath)

if __name__ == "__main__":
    main()