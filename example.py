import numpy as np

from mlp import MultilayerPerceptron
from utils import load_mnist


if __name__ == "__main__":
    x_train, y_train = load_mnist('', kind='mnist_data/train')
    x_test, y_test = load_mnist('', kind='mnist_data/t10k')

    nn = MultilayerPerceptron(layers_hidden=(100, ), epochs=100, eta=0.0005,
                              shuffle=True, batch=100, seed=1)
    nn.fit(x_train[:55000], y_train[:55000], x_train[55000:], y_train[55000:])

    y_test_pred = nn.predict(x_test)
    acc = np.sum(y_test == y_test_pred).astype(np.float) / x_test.shape[0]
    print('accuracy on test data: {:.2f}'.format(acc * 100))
