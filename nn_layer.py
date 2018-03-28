import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_delta(a):
    return sigmoid(a)*(1 - sigmoid(a))

def relu(a):
    return np.maximum(0., a)

def relu_delta(a):
    return np.maximum(np.minimum(1, np.round(a + 0.5)), 0)

def softmax(a):
    tmp = np.exp(a - np.max(a, axis=0))
    return tmp/np.sum(tmp, axis=0)

def tanh(a):
    return np.tanh(a)

def tanh_delta(a):
    return 1 - tanh(a) ** 2

def linear(a):
    return a

def linear_delta(a):
    return 1.

class Layer:
    def __init__(self, output_shape, activation='linear', LR=0.01):
        self.output_shape = output_shape
        self.eta = LR
        self.skip_act_delta = False
        if activation == 'relu':
            self.act = relu
            self.act_delta = relu_delta
        elif activation == 'sigmoid':
            self.act = sigmoid
            self.act_delta = sigmoid_delta
        elif activation == 'softmax':
            self.act = softmax
            self.act_delta = None
            self.skip_act_delta = True
        elif activation == 'tanh':
            self.act = tanh
            self.act_delta = tanh_delta
        else:
            self.act = linear
            self.act_delta = linear_delta

    def set_shape(self, input_shape):
        self.input_shape = input_shape
        self.w = np.random.randn(self.output_shape, input_shape + 1)

    def forward_prop(self, x):
        self.h = np.row_stack((x, np.ones(x.shape[1])))
        return self.act(self.w.dot(self.h))

    def backword_prop(self, g):
        if not self.skip_act_delta:
            g = self.act_delta(self.w.dot(self.h)) * g
        dw = g.dot(self.h.T)
        if len(g.shape) > 1:
            g = self.w[:, :-1].T.dot(g)
        else:
            g = np.outer(self.w[:-1], g)
        self.w -= self.eta * dw
        return g

    def get_hidden_state(self):
        return self.h[:-1, :]

    def get_output_shape(self):
        return self.output_shape

def model_configure(model, input_shape):
    shape = input_shape
    decription = str(input_shape)
    for layer in model:
        layer.set_shape(shape)
        shape = layer.get_output_shape()
        decription = decription +  " - " + str(shape)
    return model, decription

def predict(model, x):
    h = x
    for layer in model:
        h = layer.forward_prop(h)
    return h