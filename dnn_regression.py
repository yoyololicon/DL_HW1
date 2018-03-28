import numpy as np
import argparse
import matplotlib.pyplot as plt

from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from prettytable import PrettyTable

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
    return np.minimum(1, np.round(r + 0.5))

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

def training(train_x, test_x, train_t, test_t, depth, epochs, batch_size, node_n, eta, f_names, activation='relu'):
    train_size = train_x.shape[1]
    iw = np.random.randn(node_n, train_x.shape[0]+1)
    ow = np.random.randn(node_n + 1)
    w = np.random.randn(depth - 3, node_n, node_n + 1)
    h = np.full((depth - 2, node_n + 1, batch_size), 1.)
    train_error = []
    test_error = []
    every = 1000

    framework = str(train_x.shape[0]) + "-"
    for i in range(depth-2):
        framework += str(node_n) + "-"
    framework += "1"
    table = PrettyTable(["Network architecture", framework])
    table.add_row(["Activation", activation])
    table.add_row(["Selected features", f_names])

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
        if e % every == 0:
            print "Epoch", e, ": train loss", train_error[-1]

    test_y = predict(test_x, (iw, w, ow), activation=activation)
    table.add_row(["Training RMS Error", train_error[-1]])
    table.add_row(["Test RMS Error", RMSE(test_y, test_t)])
    print table

    plt.plot(train_error, label='train loss')
    plt.plot(test_error, label='test loss')
    plt.legend()
    plt.title("Learning curve with " + str(train_x.shape[0]) + " dimension inputs")
    plt.xlabel("# of epoch")
    plt.ylabel("square loss")
    plt.show()

    train_y = predict(train_x, (iw, w, ow), activation=activation)
    plt.plot(train_y, label='y')
    plt.plot(train_t, label='t')
    plt.legend()
    plt.title("Heat load for training dataset with " + str(train_x.shape[0]) + " dimension inputs")
    plt.xlabel("# of case")
    plt.ylabel("Heat load")
    plt.show()

    plt.plot(test_y, label='y')
    plt.plot(test_t, label='t')
    plt.legend()
    plt.title("Heat load for test dataset with " + str(train_x.shape[0]) + " dimension inputs")
    plt.xlabel("# of case")
    plt.ylabel("Heat load")
    plt.show()

    return train_error


def feature_corr(data):
    feature_names = list(data.columns.values)
    feature_names.remove(target_name)
    feature_names.remove(excluded)
    y = data[target_name].values

    features = []
    for name in feature_names:
        x = data[name].values
        corr = pearsonr(x, y)
        features.append((name, abs(corr[0])))
    features.sort(key=lambda x: x[1], reverse=True)
    return features

if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_csv(args.infile)
    #shuffle data at beginning
    data = data.sample(frac=1).reset_index(drop=True)

    feature_sorted = feature_corr(data)
    t = data[target_name].values

    feature_data = []
    encoders = []
    test_size = 192
    f_names = ""
    for i in range(len(feature_sorted)):
        name = feature_sorted[i][0]
        if 0 < i < len(feature_sorted) - 1:
            f_names += ", "
        f_names += name
        if name in categorical:
            encoders.append((name, OneHotEncoder(sparse=False)))
            feature_data.append(encoders[-1][1].fit_transform(data[name].values.reshape(-1, 1)))
        else:
            feature_data.append(data[feature_sorted[i][0]].values)
        x = np.column_stack(feature_data)
        x = normalize(x, axis=0)

        train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=test_size, shuffle=False)
        train_x, test_x = train_x.T, test_x.T

        training(train_x, test_x, train_t, test_t, depth=5, epochs=30000, batch_size=192, node_n=3, eta=1e-4,
                 f_names=f_names, activation='sigmoid')
