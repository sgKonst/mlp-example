import numpy as np


class MultilayerPerceptron:

    def __init__(self, layers_hidden=None, epochs=100, eta=0.001,
                 shuffle=True, batch=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.batch = batch
        self.layers_hidden = layers_hidden or (30, )

    def predict(self, x):
        a_h, a_out = self._forward(x)
        y_pred = np.argmax(a_out, axis=1)
        return y_pred

    def fit(self, x_train, y_train, x_valid, y_valid):
        n_output = np.unique(y_train).shape[0]
        n_features = x_train.shape[1]

        # init weights
        self.w_h = []
        for layer_idx, n_hidden in enumerate(self.layers_hidden):
            size = (n_features if layer_idx == 0 else self.layers_hidden[layer_idx - 1], n_hidden)
            self.w_h.append(self.random.normal(loc=0.0, scale=0.1, size=size))

        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.layers_hidden[-1], n_output))

        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)

        for i in range(self.epochs):
            indices = np.arange(x_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for idx in range(0, indices.shape[0] - self.batch + 1, self.batch):
                batch_idx = indices[idx:idx + self.batch]
                a_h_list, a_out = self._forward(x_train[batch_idx])

                sigma_out = a_out - y_train_enc[batch_idx]

                # calculate gradients for all layers
                w_h = self.w_out
                sigma_h = sigma_out
                grad_w_h = []
                for layer_idx, a_h in list(enumerate(a_h_list))[::-1]:
                    sigmoid_derivative_h = a_h * (1. - a_h)
                    sigma_h = np.dot(sigma_h, w_h.T) * sigmoid_derivative_h
                    a_h_prev = a_h_list[layer_idx - 1] if layer_idx > 0 else x_train[batch_idx]
                    grad_w_h.append(np.dot(a_h_prev.T, sigma_h))

                    w_h = self.w_h[layer_idx]

                grad_w_out = np.dot(a_h_list[-1].T, sigma_out)
                grad_w_h = grad_w_h[::-1]

                # update weights
                for w_idx, w_h in enumerate(self.w_h):
                    self.w_h[w_idx] -= self.eta * grad_w_h[w_idx]
                self.w_out -= self.eta * grad_w_out

            # calculate cost
            a_h, a_out = self._forward(x_train)

            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(x_train)
            y_valid_pred = self.predict(x_valid)
            train_acc = np.sum(y_train == y_train_pred).astype(np.float) / x_train.shape[0]
            valid_acc = np.sum(y_valid == y_valid_pred).astype(np.float) / x_valid.shape[0]

            print('{:4d}/{:4d} | cost: {:.2f} | train/validation accuracy: {:.2f}/{:.2f}'.format(
                i, self.epochs, cost, train_acc * 100, valid_acc * 100
            ))

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

    def _onehot(self, y, n_classes):
        """
        Encode labels using OneHot method
        """
        onehot = np.zeros((n_classes, y.shape[0]))

        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.

        return onehot.T

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, x):
        a_h = []
        a_last = x
        for layer_idx, _ in enumerate(self.layers_hidden):
            z_h = np.dot(a_last, self.w_h[layer_idx])
            a_last = self._sigmoid(z_h)
            a_h.append(a_last)

        z_out = np.dot(a_last, self.w_out)
        a_out = self._sigmoid(z_out)

        return a_h, a_out

    def _compute_cost(self, y_enc, output):
        cost = np.sum(-y_enc * np.log(output) - (1. - y_enc) * np.log(1. - output))
        return cost
