import numpy as np

class Perceptron:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    # ... (Keep all the other methods unchanged)

    def train(self, inputs, labels, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagation(inputs, labels, learning_rate)
            if epoch % 10 == 0:
                loss = self.cross_entropy_loss(self.forward(inputs), labels)
                print(f'Epoch {epoch}, Loss: {loss}')
