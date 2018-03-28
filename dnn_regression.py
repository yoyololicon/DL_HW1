import numpy as np
import argparse
import matplotlib.pyplot as plt

from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from nn_layer import Layer, model_configure, predict
from scipy.stats.stats import pearsonr
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='DNN regression')
parser.add_argument('infile', help='csv file name')
target_name = 'Heating Load'
excluded = 'Cooling Load'
categorical = ['Orientation', 'Glazing Area Distribution']

def RMSE(y, t):
    return np.sqrt(np.mean((y - t) ** 2))

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
    #data = data.sample(frac=1).reset_index(drop=True)

    feature_names = list(data.columns.values)
    feature_names.remove(target_name)
    feature_names.remove(excluded)
    t = data[target_name].values

    features = []
    encoders = []
    for name in feature_names:
        if name in categorical:
            encoders.append((name, OneHotEncoder(sparse=False)))
            features.append(encoders[-1][1].fit_transform(data[name].values.reshape(-1, 1)))
        else:
            features.append(data[name].values)
    x = np.column_stack(features)
    x = normalize(x, axis=0)

    test_size = 192
    train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=test_size, shuffle=True)
    train_x, test_x = train_x.T, test_x.T

    input_dim = train_x.shape[0]
    train_size = train_x.shape[1]
    LR = 1e-4
    batch_size = 64
    epochs = 10000
    every = 1000

    model = []
    model.append(Layer(6, activation='sigmoid', LR=LR))
    model.append(Layer(3, activation='sigmoid', LR=LR))
    #model.append(Layer(2, activation='sigmoid', LR=LR))
    model.append(Layer(1, activation='linear', LR=LR))

    model, framework = model_configure(model, input_dim)
    table = PrettyTable(["Network architecture", framework])
    table.add_row(["Selected features", "all"])

    train_loss = []
    for e in range(epochs):
        batch_loss_sum = 0
        for i in range(train_size / batch_size):
            h = train_x[:, i:i + batch_size]
            batch_t = train_t[i:i + batch_size]

            # forward
            for layer in model:
                h = layer.forward_prop(h)

            batch_loss_sum += np.sum((h - batch_t) ** 2)
            gradient = h - batch_t

            model.reverse()
            # backward
            for layer in model:
                gradient = layer.backword_prop(gradient)
            model.reverse()

        train_loss.append(np.sqrt(batch_loss_sum/train_size))
        if e % every == 0:
            print "Epoch", e, ": train loss", train_loss[-1]

    test_y = np.squeeze(predict(model, test_x))
    train_y = np.squeeze(predict(model, train_x))
    table.add_row(["Training RMS Error", RMSE(train_y, train_t)])
    table.add_row(["Test RMS Error", RMSE(test_y, test_t)])
    print table

    plt.plot(train_loss)
    plt.title("Training loss")
    plt.xlabel("# of epoch")
    plt.ylabel("Squre loss")
    plt.show()

    plt.plot(train_t, label='label')
    plt.plot(train_y, label='predict')
    plt.title("Heat load for training dataset")
    plt.ylabel("Heat load")
    plt.xlabel("#th case")
    plt.legend()
    plt.show()

    plt.plot(test_t, label='label')
    plt.plot(test_y, label='predict')
    plt.title("Heat load for test dataset")
    plt.ylabel("Heat load")
    plt.xlabel("#th case")
    plt.legend()
    plt.show()
    """
    feature_sorted = feature_corr(data)
    t = data[target_name].values

    feature_data = []
    encoders = []

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
    """