"Main program"

#third-party lib
import pygame
from pygame.locals import *
import numpy as np

#std lib
import sys

#dependacies
import model
import torch
from cnn2 import SCNN
#import cnn2 as model

#const
HEIGHT = 600
WIDTH = 800
SQUARE_SIZE = 15

#get position
def getSquaredPosition(pos,translation):
    for i in range(0,2):
     if(pos[i] < translation[i] or pos[i] >= translation[i] + (28 * SQUARE_SIZE)):
         pos = [-1,-1]
         return 
     
     pos[i] -= translation[i]
     pos[i] = pos[i] - (pos[i]%SQUARE_SIZE)
     pos[i] += translation[i]

    return pos

#draw the grid
def drawGrid(renderer):
        number_of_squares = 28
        translation_x = (WIDTH//2)-((number_of_squares*SQUARE_SIZE)//2)
        translation_y = 50

        for i in range(0,(number_of_squares+1)):
          begin_vertical = ((i*SQUARE_SIZE)+translation_x,translation_y)
          end_vertical = ((i*SQUARE_SIZE)+translation_x,((number_of_squares)*SQUARE_SIZE)+translation_y)
          begin_horizontal = (translation_x,(i*SQUARE_SIZE)+translation_y)
          end_horizontal = (((number_of_squares)*SQUARE_SIZE) + translation_x,(i*SQUARE_SIZE)+translation_y)

          pygame.draw.line(renderer,(0,0,0),(begin_horizontal),(end_horizontal),1)
          pygame.draw.line(renderer,(0,0,0),(begin_vertical),(end_vertical),1)

          """if i == 1:
              i = number_of_squares 
          begin_vertical = ((i*SQUARE_SIZE)+translation_x,translation_y)
          end_vertical = ((i*SQUARE_SIZE)+translation_x,((number_of_squares)*SQUARE_SIZE)+translation_y)
          begin_horizontal = (translation_x,(i*SQUARE_SIZE)+translation_y)
          end_horizontal = (((number_of_squares)*SQUARE_SIZE) + translation_x,(i*SQUARE_SIZE)+translation_y)
          pygame.draw.line(renderer,(0,0,0),begin_horizontal,end_horizontal,1)
          pygame.draw.line(renderer,(0,0,0),begin_vertical,end_vertical,1)"""

        translation = [translation_x,translation_y]
        return translation

#find the square
def findSquare(pos,squares):
    pos_ = []
    for rect in squares:
        pos_ = [rect.x,rect.y]
        if (pos_ == pos):
            squares.remove(rect)

#create the image for the neural network to recognize
def create_input(pixels,squares,translation):
    for square in squares:
        pos = [square.left,square.top]
        for i in range(2):
            pos[i] -= translation[i]
        row = pos[0] // SQUARE_SIZE
        column = pos[1] // SQUARE_SIZE
        pixel = row + (28 * column)
        pixels[pixel] = 1.0
    return pixels


#main function
def run(network, mode, model_type):
    #init pygame
    pygame.init()

    #init renderer
    renderer = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("Hand Write")

    #init squares
    squares = []

    #load the neural network
    #network = model.load_network("nn_model.json",1)
    #state_dict = torch.load("scnn_model.pth",weights_only=False)

    #the canvas you drew
    pixels = []
    if model_type == 0:
        pixels = np.zeros((784,1))
    else:
       pixels = torch.zeros((784,1))

    #loop
    while True:
     for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                #squares = []
                pixels = create_input(pixels,squares,translation)
                digit = -1
                if model_type == 0:
                  output = network.forward_propagation(pixels)
                  digit = np.argmax(output)
                  pixels = np.zeros((784,1))

                else:
                   digit = torch.argmax(output)
                   pixels = torch.reshape(pixels,[1,1,28,28])
                   output = network.forward_propagation(pixels)
                   pixels = torch.zeros((784,1))
                print("Prediction: " + str(digit))
                print(output)
            if event.key == pygame.K_BACKSPACE:
                squares = []
            
    
     renderer.fill((255,255,255))
     translation  = drawGrid(renderer)
     mouseEvents = pygame.mouse.get_pressed()

     if(mouseEvents[0] == True):
        mousePos = pygame.mouse.get_pos()
        mousePos = getSquaredPosition(list(mousePos),translation)
        if(mousePos != None):
            startPos = [mousePos[0] - (SQUARE_SIZE), mousePos[1] - (SQUARE_SIZE)]

            if(mousePos[0] != -1): 
              #squares.append(pygame.Rect(mousePos[0],mousePos[1],SQUARE_SIZE,SQUARE_SIZE))
              for i in range(9):
                x = i % 3
                y = i // 3
                pos = [startPos[0] + (x*SQUARE_SIZE), startPos[1] + (y*SQUARE_SIZE)]
                if(
                    pos[0] >= translation[0] and
                    pos[0] < translation[0] + (SQUARE_SIZE * 28) and
                    pos[1] >= translation[1] and
                    pos[1] < translation[1] + (SQUARE_SIZE * 28)
                ):
                    squares.append(pygame.Rect(pos[0],pos[1],SQUARE_SIZE,SQUARE_SIZE))

     elif(mouseEvents[2] == True):
        mousePos = pygame.mouse.get_pos()
        mousePos = getSquaredPosition(list(mousePos),translation)

        if(mousePos[0] != -1): 
            findSquare(mousePos,squares)

     for square in squares:
        pygame.draw.rect(renderer,(0,0,0),square,0)
     pygame.display.update()

menu = [
    "Main Menu",
    "0: Quit",
    "1: Model",
    "2: Test",
    "3: Run"
]

models_menu = [
    "Choose your model:",
    "1: Neural Network (NN)",
    "2: Convolutional Neural Network (CNN)",
    "3: Superior Convolutional Neural Network (SCNN)",
    "4: Superior Grayscale Convolutional Neural Network (SCNN)"
    "0: Go Back -\__:-:_/-"
]

models_name = [
    "Neural Network (NN)",
    "Convolutional Neural Network (CNN)",
    "Superior Convolutional Neural Network (SCNN)",
    "Superior Grayscale Convolutional Neural Network (SCNN)"
    "Nothing :("
]

model_filepaths = [
    "nn_model.json",
    "cnn_model.json",
    "scnn_model.pth",
    "scnn_model_2.pth"
]

def main():
    network = None
    menu_command = -1
    model_command = -1
    current_model = -1
    mode = -1
    #if mode == 0, we are just running
    #if mode == 1, we are testing
    model_type = -1
    #if model_type == 0: Numpy
    #if model_type == 1: Pytorch

    while True:
        print("We are currently using: ")
        print("We are using : ")
        current_model = model_command
        if current_model != -1:
            current_model = model_command - 1 
        print(models_name[current_model])
        if(menu_command != 1):
            model.print_(menu)
            menu_command = model.test_NAN(menu_command,"What's your command: ",menu)
            model.number_expectation(menu,0,4,menu_command)
        #print(command)

        if menu_command == 0:
            print("Bye...")
            break

        elif menu_command == 1:
            choose_model = True
            while choose_model == True:
                model.print_(models_menu)
                model_command = model.test_NAN(model_command,"Whats's your command: ",models_menu)
                model.number_expectation(models_menu,0,2,model_command)
                
                if model_command == 0:
                    print("Bye")
                    choose_model = False
                    menu_command = -1
                elif model_command == 1:
                    network = model.load_network(model_filepaths[0],1)
                    print("The model in usage is the Multilayer Perceptron / Neural Network (NN) ")
                    model_type = 0
                    choose_model = False
                    menu_command = -1
                elif model_command == 2:
                    network = model.load_network(model_filepaths[1],2)
                    print("The model in usage is the Multilayer Perceptron + Convolutional Neural Network (CNN) ")
                    choose_model = False
                    menu_command = -1
                    model_type = 0
                elif model_command == 3:
                    network = torch.load(model_filepaths[2],weights_only=False)
                    network.eval()
                    print("The model in usage is the Superior Multilayer Perceptron + Convolutional Neural Network (SCNN) ")
                    choose_model = False
                    menu_command = -1
                    model_type = 1
                elif model_command == 4:
                    network = torch.load(model_filepaths[3],weights_only=False)
                    network.eval()
                    print("The model in usage is the Superior Multilayer Perceptron + Convolutional Neural Network (SCNN) ")
                    choose_model = False
                    menu_command = -1
                    model_type = 1
        
        elif menu_command == 2:
            mode = 1
            run(network,mode,model_type)

        elif menu_command == 3:
            mode = 0
            run(network,mode,model_type)

    
if __name__ == "__main__":
    main()