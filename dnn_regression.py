import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from nn_layer import *
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='DNN regression')
parser.add_argument('infile', help='csv file name')
target_name = 'Heating Load'
excluded = 'Cooling Load'
categorical = ['Orientation', 'Glazing Area Distribution']

if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_csv(args.infile)

    feature_names = list(data.columns.values)
    feature_names.remove(target_name)
    feature_names.remove(excluded)
    for c in categorical:
        feature_names.remove(c)

    features = []
    for name in feature_names:
        features.append(data[name].values)

    non_category = len(features)
    encoders = []
    for name in categorical:
        encoders.append((name, OneHotEncoder(sparse=False)))
        features.append(encoders[-1][1].fit_transform(data[name].values.reshape(-1, 1)))

    t = data[target_name].values
    x = np.column_stack(features)
    x[:, :non_category] = normalize(x[:, :non_category], axis=0)

    test_size = 192
    train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=test_size, shuffle=True)
    train_x, test_x = train_x.T, test_x.T

    input_dim = train_x.shape[0]
    train_size = train_x.shape[1]
    LR = 0.01
    batch_size = 192
    epochs = 20000
    every = 1000

    model = []
    model.append(Layer(6, activation='sigmoid', LR=LR))
    model.append(Layer(3, activation='sigmoid', LR=LR))
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

            batch_y = predict(model, h)

            batch_loss_sum += np.sum((batch_y - batch_t) ** 2)
            gradient = batch_y - batch_t

            back_propogation(model, gradient)

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

    default_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = [12, 9]
    fig, sub = plt.subplots(2)
    sub[0].plot(train_t, label='label')
    sub[0].plot(train_y, label='predict')
    sub[0].set_title("Heat load for training dataset")
    sub[0].set_ylabel("Heat load")
    sub[0].legend()

    sub[1].plot(test_t, label='label')
    sub[1].plot(test_y, label='predict')
    sub[1].set_title("Heat load for test dataset")
    sub[1].set_ylabel("Heat load")
    sub[1].set_xlabel("#th case")
    sub[1].legend()
    plt.show()

    noise_test = np.random.randn(test_x.shape[1])*0.25
    noise_train = np.random.randn(train_x.shape[1])*0.25

    print "Add random noise to each non-category features in test data;"
    print "then evaluate the RMS error on the trained model."
    table = PrettyTable(["Noisy feature", "Test RMS error"])

    for i, name in enumerate(feature_names):
        print i
        test_x[i, :] += noise_test
        test_y = np.squeeze(predict(model, test_x))
        table.add_row([name, RMSE(test_y, test_t)])
        test_x[i, :] -= noise_test

    print table

    print "Add random noise to each non-category features in training data;"
    print "then re-trained the model."

    for i, name in enumerate(feature_names):
        table = PrettyTable(["Network architecture", framework])
        table.add_row(["Noisy feature", name])

        train_x[i, :] += noise_train
        initialize(model)
        train_loss = []
        for e in range(epochs):
            batch_loss_sum = 0
            for i in range(train_size / batch_size):
                h = train_x[:, i:i + batch_size]
                batch_t = train_t[i:i + batch_size]

                batch_y = predict(model, h)

                batch_loss_sum += np.sum((batch_y - batch_t) ** 2)
                gradient = batch_y - batch_t

                back_propogation(model, gradient)

            train_loss.append(np.sqrt(batch_loss_sum/train_size))

        test_y = np.squeeze(predict(model, test_x))
        train_y = np.squeeze(predict(model, train_x))
        table.add_row(["Training RMS Error", RMSE(train_y, train_t)])
        table.add_row(["Test RMS Error", RMSE(test_y, test_t)])
        print table

        plt.rcParams["figure.figsize"] = default_size
        plt.plot(train_loss)
        plt.title("Training loss with noisy " + name)
        plt.xlabel("# of epoch")
        plt.ylabel("Squre loss")
        plt.show()

        plt.rcParams["figure.figsize"] = [12, 9]
        fig, sub = plt.subplots(2)
        sub[0].plot(train_t, label='label')
        sub[0].plot(train_y, label='predict')
        sub[0].set_title("Heat load for training dataset with noisy " + name)
        sub[0].set_ylabel("Heat load")
        sub[0].legend()

        sub[1].plot(test_t, label='label')
        sub[1].plot(test_y, label='predict')
        sub[1].set_title("Heat load for test dataset with noisy " + name)
        sub[1].set_ylabel("Heat load")
        sub[1].set_xlabel("#th case")
        sub[1].legend()
        plt.show()

        train_x[i, :] -= noise_train
