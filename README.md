# HandWrite
Hand write a number (for the moment) and it recognizes it (kinda)

# Description
I used MNSIT data for training. It was a collection of handwritten digits (ranging from 0 to 9). I had 60 000 images as training data and 10 000 images as test data. Each image have a label.

I wrote 4 models:
- NN (Simple Neural Network) from scratch with NumPy
- CNN (Convolutional Neural Network) from scratch with NumPy
- SCNN 1 (CNN trained with the help of binary data) with PyTorch
- SCNN 2 (CNN trained with the help of grayscale data) with PyTorch

# Neural Network
For the Neural Network, I used NumPy. For this model, I used ReLU as the nonlinear activation function. For the last neuron, I used softmax as my normalization function. As for the optimization, SGD was used for this model. It was a very simple NN that could reach 99% during training with 0.01 as a learning rate, 60 images for  each batch and 16 epochs. But its accuracy with the drawing interface is barely 49%

# Convolutional Neural Network
For the Convolutional Neural Network, I used NumPy. I made the filter calculation with NumPy, but this is not fast enough. In the convolutional layer, you can as much filter as you want. In the next step, I use a maxpool to get the "feature". This convolutional step is slow due to the fact that it wasn't "optimized" with better linear algebra. But, for the multi layer perceptron, I used the same algorithm as the Neural Network