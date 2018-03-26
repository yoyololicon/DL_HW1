import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='DNN regression')
parser.add_argument('infile', help='csv file name')
target_name = 'Heating Load'
excluded = 'Cooling Load'
categorical = ['Orientation', 'Glazing Area Distribution']

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_dif(sig):
    return sig*(1 - sig)

def relu(a):
    return np.maximum(0., a)

def relu_dif(r):
    r2 = np.round(r + 0.5)
    return np.minimum(1, r2)

def RMSE(y, t):
    return np.sqrt(np.mean((y - t) ** 2))

def predict(x, weights, activation='relu'):
    (iw, w, ow) = weights
    test_len = x.shape[1]
    x2 = np.row_stack((x, np.ones(test_len)))
    if activation == 'sigmoid':
        act = sigmoid
    else:
        act = relu
    h = np.row_stack((act(iw.dot(x2)), np.ones(test_len)))
    for i in range(w.shape[0]):
        h = np.row_stack((act(w[i].dot(h)), np.ones(test_len)))
    return ow.dot(h)

def training(x, t, depth, epochs, batch_size, test_size, node_n, eta, activation='relu'):
    train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=test_size, shuffle=True)
    train_x, test_x = train_x.T, test_x.T

    # DNN
    train_size = train_x.shape[1]
    iw = np.random.rand(node_n, train_x.shape[0]+1)
    ow = np.random.rand(node_n + 1)
    w = np.random.rand(depth - 3, node_n, node_n + 1)
    h = np.full((depth - 2, node_n + 1, batch_size), 1.)
    train_error = []
    test_error = []
    every = 1000

    if activation == 'sigmoid':
        act = sigmoid
        act_dif = sigmoid_dif
    else:
        act = relu
        act_dif = relu_dif

    # SGD
    for e in range(epochs):
        batch_error_sum = 0
        for i in range(train_size / batch_size):
            batch_x = np.row_stack((train_x[:, i:i + batch_size], np.ones(batch_size)))
            batch_t = train_t[i:i + batch_size]

            # forward
            for k in range(1, depth):
                if k == depth - 1:
                    batch_y = ow.dot(h[k - 2])
                elif k == 1:
                    h[k - 1, :-1] = act(iw.dot(batch_x))
                else:
                    h[k - 1, :-1] = act(w[k - 2].dot(h[k - 2]))

            batch_error_sum += np.sum((batch_y - batch_t) ** 2)
            g = batch_y - batch_t
            # backward
            for k in range(depth - 1, 0, -1):
                if k == depth - 1:
                    dw = g.dot(h[k - 2].T)
                    g = np.outer(ow[:-1], g)
                    ow -= eta * dw
                elif k == 1:
                    g = act_dif(h[k - 1, :-1])*g
                    dw = g.dot(batch_x.T)
                    iw -= eta * dw
                else:
                    g = act_dif(h[k - 1, :-1]) * g
                    dw = g.dot(h[k - 2].T)
                    g = w[k - 2, :, :-1].T.dot(g)
                    w[k - 2] -= eta * dw

        train_error.append(np.sqrt(batch_error_sum / train_size))
        test_y = predict(test_x, (iw, w, ow), activation=activation)
        test_error.append(RMSE(test_y, test_t))
        if e % every == 0:
            print "Epoch", e, ": train loss", train_error[-1], "; test loss", test_error[-1]

    plt.plot(test_y, label='y')
    plt.plot(test_t, label='t')
    plt.legend()
    plt.show()

    plt.plot(train_error, label='train loss')
    plt.plot(test_error, label='test loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_csv(args.infile)

    x = []
    encoders = []
    for name in data.columns:
        if name == target_name:
            t = data[name].values
        elif name in excluded:
            continue
        elif name in categorical:
            encoders.append(OneHotEncoder(sparse=False))
            x.append(encoders[-1].fit_transform(data[name].values.reshape(-1, 1)))
        else:
            x.append(data[name].values)

    x = np.column_stack(x)

    training(x, t, depth=5, epochs=100000, batch_size=64, test_size=192, node_n=2, eta=1e-8, activation='relu')
