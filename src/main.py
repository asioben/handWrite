"Main program"

#third-party lib
import pygame
from pygame.locals import *
import numpy as np

#std lib
import sys

#dependacies
import nn

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
def main():
    #init pygame
    pygame.init()

    #init renderer
    renderer = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("Hand Write")

    #init squares
    squares = []

    #load the neural network
    network = nn.load_network("model.json")

    #the canvas you drew
    pixels = np.zeros((784,1))

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
                output = network.forward_propagation(pixels)
                digit = np.argmax(output)
                print("Prediction: " + str(digit))
                #print("True: " + str(model.load.y_test[6020]))
                #print(pixels)
                #print("/////////////////////////////")
                print(output)
            if event.key == pygame.K_BACKSPACE:
                squares = []
            
    
     renderer.fill((255,255,255))
     translation  = drawGrid(renderer)
     mouseEvents = pygame.mouse.get_pressed()

     if(mouseEvents[0] == True):
        mousePos = pygame.mouse.get_pos()
        mousePos = getSquaredPosition(list(mousePos),translation)
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
                    print(":"+ str(translation[0]) + "," + str(pos[1]))
                    squares.append(pygame.Rect(pos[0],pos[1],SQUARE_SIZE,SQUARE_SIZE))

     elif(mouseEvents[2] == True):
        mousePos = pygame.mouse.get_pos()
        mousePos = getSquaredPosition(list(mousePos),translation)

        if(mousePos[0] != -1): 
            findSquare(mousePos,squares)

     for square in squares:
        pygame.draw.rect(renderer,(0,0,0),square,0)
     pygame.display.update()
    
if __name__ == "__main__":
    main()