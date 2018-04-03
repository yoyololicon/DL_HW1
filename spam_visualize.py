import argparse
from scipy.io import loadmat
from nn_layer import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description='spam email dataset training latent features visualization')
parser.add_argument('infile', help='matlab file name')
parser.add_argument('-n', type=int, choices=range(2, 4))

if __name__ == '__main__':
    args = parser.parse_args()
    data = loadmat(args.infile)
    train_x, train_t, test_x, test_t = data['train_x'].T, data['train_y'].T, data['test_x'].T, data['test_y'].T
    input_dim = train_x.shape[0]
    output_dim = train_t.shape[0]
    train_size = train_x.shape[1]

    LR = 0.1
    batch_size = 400
    epochs = 600
    every = 100

    model = []

    model.append(Layer(20, activation='sigmoid', LR=LR))
    model.append(Layer(10, activation='tanh', LR=LR))
    model.append(Layer(5, activation='sigmoid', LR=LR))
    model.append(Layer(args.n, activation='linear', LR=LR))
    model.append(Layer(output_dim, activation='softmax', LR=LR))

    model, framework = model_configure(model, input_dim)

    for e in range(epochs):
        for i in range(train_size / batch_size):
            h = train_x[:, i:i + batch_size]
            batch_t = train_t[:, i:i + batch_size]

            batch_y = predict(model, h)
            gradient = batch_y - batch_t
            back_propogation(model, gradient)

        if e % every == 0:
            #plot distribution
            latent = model[-1].get_hidden_state()
            if args.n == 2:
                plt.scatter(latent[0], latent[1], c=batch_t[0], cmap='coolwarm', alpha=0.8)
                plt.title(str(args.n)+"D features after "+str(e)+" epoch")
                plt.show()
            else:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(latent[0], latent[1], latent[2], c=batch_t[0], cmap='coolwarm', alpha=0.8)
                ax.set_title(str(args.n)+"D features after "+str(e)+" epoch")
                plt.show()