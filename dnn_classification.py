import argparse
from scipy.io import loadmat
from sklearn.metrics import log_loss, zero_one_loss
from sklearn.preprocessing import normalize
from nn_layer import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='DNN classification')
parser.add_argument('infile', help='matlab file name')

if __name__ == '__main__':
    args = parser.parse_args()
    data = loadmat(args.infile)
    train_x, train_t, test_x, test_t = data['train_x'].T, data['train_y'].T, data['test_x'].T, data['test_y'].T
    """
    x_concat = np.concatenate((train_x, test_x), axis=1)
    x_concat = normalize(x_concat)
    train_x, test_x = np.split(x_concat, [train_x.shape[1]], axis=1)
    """
    input_dim = train_x.shape[0]
    output_dim = train_t.shape[0]
    train_size = train_x.shape[1]

    LR = 0.1
    batch_size = 400
    epochs = 2000
    every = 100

    model = []

    model.append(Layer(20, activation='sigmoid', LR=LR))
    model.append(Layer(10, activation='tanh', LR=LR))
    model.append(Layer(5, activation='sigmoid', LR=LR))
    model.append(Layer(output_dim, activation='softmax', LR=LR))

    model, framework = model_configure(model, input_dim)
    table = PrettyTable(["Network architecture", framework])

    train_loss = []
    train_error = []
    test_error = []
    for e in range(epochs):
        batch_loss_sum = 0
        batch_error_rate = 0
        for i in range(train_size / batch_size):
            h = train_x[:, i:i + batch_size]
            batch_t = train_t[:, i:i + batch_size]

            batch_y = predict(model, h)

            batch_loss_sum += log_loss(batch_t.T, batch_y.T)
            batch_error_rate += zero_one_loss(batch_t.T, batch_y.T.round())
            gradient = batch_y - batch_t

            back_propogation(model, gradient)

        train_loss.append(batch_loss_sum)
        train_error.append(batch_error_rate/(train_size/batch_size))
        test_y = predict(model, test_x)
        test_error.append(zero_one_loss(test_t.T, test_y.T.round()))
        if e % every == 0:
            print "Epoch", e, ": train loss", train_loss[-1]

    table.add_row(["Training loss", train_loss[-1]])
    table.add_row(["Training Error", train_error[-1]])
    table.add_row(["Test Error", test_error[-1]])
    print table

    plt.plot(train_loss, label='train loss')
    plt.title("Training loss")
    plt.xlabel("# of epoch")
    plt.ylabel("Cross entropy")
    plt.show()

    plt.plot(train_error, label='train error')
    plt.plot(test_error, label='test error')
    plt.title(framework)
    plt.xlabel("# of epoch")
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()

