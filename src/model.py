"Model of Neural Network 1.0"

#linking
import load
import nn

#std lib
import os 

#data to train and test the model
x_train = load.x_train
y_train = load.y_train
x_test = load.x_test
y_test = load.y_test

#functions for the panel of control
errorCode = "Invalid input !"

def print_menu():
    menu = [
        "Menu:",
        "0: Quit",
        "1: Train",
        "2: Predict",
        "3: Download"
    ]
    for text in menu:
        print(text)

def test_NAN(test,text):
    while True:
        try:
            test = int(input(text))
            return test
        except ValueError:
            print(errorCode)
            print_menu()

def check_file_and_write(filepath):
    print("I'll create a new file, model.json")
    with open(filepath, "w") as file:
        file.write("")

def main():
    model = nn.NeuralNetwork((784,128,64,32,10))
    command = -1
    filepath = ""
    trained = False
    while True:
        print_menu()
        command = test_NAN(command,"What's your command: ")
        while command > 3 or command < 0:
            print(errorCode)
            print_menu()
            command = test_NAN(command,"What's your command: ")
        if command == 0:
            break
        elif command == 1:
            print("Model in training...")
            trained = True
            model.train(x_train,y_train,0.01,60,16)
        elif command == 2:
            print("We gonna predict...")
            if(trained == False):
                model = nn.load_network("model.json")
                trained = True
            model.predict(x_test,y_test)
        elif command == 3:
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