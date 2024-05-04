setup:
https://pytorch.org/get-started/locally/


tutorial:
https://towardsdatascience.com/the-math-behind-convolutional-neural-networks-6aed775df076


topics:
- datasets: training, validation, test

- topographic
    - convolutional layer
    - fully conected

- weight initializations

- Activation functions
    ideal: non linearity(better for modeling real world problems), differentiable(for backpropagation)
    - relu
        - dying neurons (if negative, derivative is 0 => therefore no learning)
    - leakingRelu
        - fast for computing, ideal for hidden layers
    - Sigmoid/tanh
        - vanishing gradient for large and small values => slow learning
        - can be used in output layer to create values in 0 to 1 / -1 to 1 range
    - softmax
        - ideal for multiclass classification

- pooling
    - reduce dimesionality (keep one value(eg max or avg) for a number of pixels)
- flattening
    - convert multidimensional tensors to vectors/single row tensor, is used before going to fully connected networks