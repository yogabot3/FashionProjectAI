import numpy as np
import pandas as pd
import os
from PIL import Image
import h5py as h5
from scipy import signal
import os


class ITrainable:

    def forward_propagation(self, X):
        raise NotImplementedError(
            "forward_propagation not implemented: ITrainable is an interface"
        )

    def backward_propagation(self, dY_hat):
        raise NotImplementedError(
            "backward_propagation not implemented: ITrainable is an interface"
        )

    def update_parameters(self):
        raise NotImplementedError(
            "update_parameters not implemented: ITrainable is an interface"
        )

    def save_parameters():
        raise NotImplementedError(
            "save_parameters not implemented: ITrainable is an interface"
        )

    def load_parameters():
        raise NotImplementedError(
            "load_parameters not implemented: ITrainable is an interface"
        )


class Network(ITrainable):
    def __init__(self, network=[]):
        self.layers = network
        self.name = "Yogev_s Convolutional network"

    def __str__(self):
        s = f"Network: \n"
        j = 0
        for i in self.layers:
            j = j + 1
            s += f"{j}. {i}\n"
        return s

    def add(self, iTrainable):
        self.layers.append(iTrainable)

    def forward_propagation(self, X):
        p = X
        for i in self.layers:
            p = i.forward_propagation(p)
        return p

    def backward_propagation(self, dY_hat):
        p = dY_hat
        for i in reversed(self.layers):
            p = i.backward_propagation(p)
        return p

    def update_parameters(self, optimization):
        for layer in self.layers:
            layer.update_parameters(optimization)

    def save_parameters(self, dir_path):
        path = dir_path + "/" + self.name
        if os.path.exists(path) == False:
            os.mkdir(path)
        for layer in self.layers:
            layer.save_parameters(path)

    def load_parameters(self, dir_path):
        path = dir_path + "/" + self.name
        for layer in self.layers:
            layer.load_parameters(path)


class Reshape(ITrainable):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        s = f"Reshape layer: \n"
        s += f"\told shape: {self.input_shape}\n"
        s += f"\tnew shape: {self.output_shape}\n"
        return s

    def forward_propagation(self, input):
        return np.reshape(input, self.output_shape)

    def backward_propagation(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)

    def update_parameters(self, optimization):
        pass

    def save_parameters(self, file_path):
        pass

    def load_parameters(self, file_path):
        pass


class ConvolutionalLayer(ITrainable):

    def __init__(self, input_shape, kernel_size, num_units, alpha, name):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.name = name
        self.input_depth = input_depth
        self.alpha = alpha
        self.num_units = num_units
        self.num_units = num_units
        self.kernels_shape = (num_units, input_depth, kernel_size, kernel_size)
        self.output_shape = (
            num_units,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernels = np.random.randn(*self.kernels_shape)
        self.b = np.random.randn(*self.output_shape)

    def forward_propagation(self, prev_A):
        self.prev_A = prev_A
        self.output = np.copy(self.b)
        for i in range(self.num_units):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.prev_A[j], self.kernels[i, j], "valid"
                )
        return self.output

    def backward_propagation(self, dZ):
        self.dZ = np.copy(dZ)
        self.kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.num_units):
            for j in range(self.input_depth):
                self.kernels_gradient[i, j] = signal.correlate(
                    self.prev_A[j], dZ[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    dZ[i], self.kernels[i, j], "full"
                )

        return input_gradient

    def update_parameters(self, optimization):
        self.kernels -= self.alpha * self.kernels_gradient
        self.b -= self.alpha * self.dZ

    def __str__(self):
        s = f"Convolutional Layer:\n"
        s += f"\tlearning_rate (alpha): {self.alpha}\n"
        s += f"\tinput shape: {self.input_shape}\n"
        s += f"\tnum units: {self.num_units}\n"
        if self.optimization != None:
            s += f"\tOptimization: {self.optimization}\n"
            if self.optimization == "adaptive":
                s += f"\t\tadaptive parameters:\n"
                s += f"\t\t\tcont: {self.adaptive_cont}\n"
                s += f"\t\t\tswitch: {self.adaptive_switch}\n"
        s += "\tParameters shape:\n"
        s += f"\t\tkernels shape: {self.kernels.shape}\n"
        s += f"\t\tb shape: {self.b.shape}\n"
        return s

    def save_parameters(self, file_path):
        kernels_flattened = self.kernels.reshape(self.kernels.shape[0], -1)
        b_flattened = self.b.reshape(self.b.shape[0], -1)
        file_name = file_path+"/"+self.name+".h5"
        with h5.File(file_name, 'w') as hf:
            hf.create_dataset("K", data=kernels_flattened)
            hf.create_dataset("b", data=b_flattened)

    def load_parameters(self, file_path):
        file_name = file_path+"/"+self.name+".h5"
        with h5.File(file_name, 'r') as hf:
            kernels_flattened = hf['K'][:]
            b_flattened = hf['b'][:]
        self.kernels = kernels_flattened.reshape(self.kernels_shape)
        self.b = b_flattened.reshape(self.output_shape)


class DenseLayer(ITrainable):

    def __init__(self, input_size, layer_size, alpha, name):
        self.alpha = alpha
        self.name = name
        self.layer_size = layer_size
        self.input_size = input_size
        self.W = np.random.randn(layer_size, input_size) * np.sqrt(2 / input_size)
        self.b = np.random.randn(layer_size, 1)
        self.adaptive_cont = 1.1
        self.adaptive_switch = 0.5
        self.adaptive_W = np.ones_like(self.W) * alpha
        self.adaptive_b = np.ones_like(self.b) * alpha

    def __str__(self):
        s = f"Dense Layer:\n"
        s += f"\tlearning_rate (alpha): {self.alpha}\n"
        s += f"\tnum inputs: {self.input_size}\n"
        s += f"\tnum units: {self.layer_size}\n"
        if self.optimization != None:
            s += f"\tOptimization: {self.optimization}\n"
            if self.optimization == "adaptive":
                s += f"\t\tadaptive parameters:\n"
                s += f"\t\t\tcont: {self.adaptive_cont}\n"
                s += f"\t\t\tswitch: {self.adaptive_switch}\n"
        s += "\tParameters shape:\n"
        s += f"\t\tW shape: {self.W.shape}\n"
        s += f"\t\tb shape: {self.b.shape}\n"
        return s

    def forward_propagation(self, prev_A):
        self.prev_A = prev_A
        Z = np.dot(self.W, self.prev_A) + self.b
        return Z

    def backward_propagation(self, dZ):
        self.dW = np.dot(dZ, self.prev_A.T)  # Normalize by batch size
        self.dZ = np.copy(dZ)
        input_gradient = np.dot(self.W.T, self.dZ)

        return input_gradient

    def update_parameters(self, optimization):
        if optimization == "adaptive":
            # Update adaptive parameters for weights
            self.adaptive_W += self.dW**2
            # Update weights using adaptive learning rate
            self.W -= self.alpha / np.sqrt(self.adaptive_W + 1e-8) * self.dW

            # Update adaptive parameters for biases
            self.adaptive_b += self.dZ**2
            # Update biases using adaptive learning rate
            self.b -= self.alpha / np.sqrt(self.adaptive_b + 1e-8) * self.dZ
        else:
            # Standard update without adaptive optimization
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.dZ

    def save_parameters(self, file_path):
        file_name = file_path + "/" + self.name + ".h5"
        with h5.File(file_name, "w") as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)

    def load_parameters(self, file_path):
        file_name = file_path + "/" + self.name + ".h5"
        with h5.File(file_name, "r") as hf:
            self.W = hf["W"][:]
            self.b = hf["b"][:]


class Activation(ITrainable):
    def __init__(self, activation):
        self.name = activation
        self.leaky_relu_d = 0.01
        if activation == "sigmoid":
            self.forward_propagation = self.sigmoid
            self.backward_propagation = self.sigmoid_dZ
        elif activation == "tanh":
            self.forward_propagation = self.tanh
            self.backward_propagation = self.tanh_dZ
        elif activation == "relu":
            self.forward_propagation = self.relu
            self.backward_propagation = self.relu_dZ
        elif activation == "leaky_relu":
            self.forward_propagation = self.leaky_relu
            self.backward_propagation = self.leaky_relu_dZ
        elif activation == "softmax":
            self.forward_propagation = self.softmax
            self.backward_propagation = self.softmax_dZ
        else:
            raise ValueError(f"{activation} is not a valid activation function\n")

    def __str__(self):
        s = f"Activation function: {self.name}\n"
        if self.name == "leaky_relu":
            s += f"\td = {self.leaky_relu_d}\n"
        return s

    def sigmoid(self, input):
        self.input = input
        Sig = 1 / (1 + np.exp(-input))
        return Sig

    def sigmoid_dZ(self, dA):
        sig = self.sigmoid(self.input)
        return np.multiply(dA, sig * (1 - sig))

    def tanh(self, Z):
        self.Z = Z
        self.A = np.tanh(Z)
        return self.A

    def tanh_dZ(self, dA):
        dA_dZ = 1 - np.tanh(self.Z) ** 2
        return dA_dZ * dA

    def relu(self, Z):
        self.input = Z
        self.res = np.maximum(0, Z)
        return self.res

    def relu_dZ(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = 0
        return dZ

    def leaky_relu(self, Z):
        self.Z = Z
        self.A = np.where(self.Z <= 0, self.leaky_relu_d * self.Z, self.Z)
        return self.A

    def leaky_relu_dZ(self, dA):
        dZ = np.where(self.Z <= 0, self.leaky_relu_d, 1)
        return dZ

    def softmax(self, Z):
        self.input = Z
        Z_max = np.max(Z, axis=0, keepdims=True)  # Subtract max for numerical stability
        exp_Z = np.exp(Z - Z_max)
        sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
        self.res = exp_Z / sum_exp_Z
        return self.res

    def softmax_dZ(self, dA):
        # Self.res is the output from the forward pass (probabilities)
        # dA is the gradient from the subsequent layer
        s = self.res.reshape(-1, 1)
        dZ = s * (dA - np.sum(dA * s, axis=0, keepdims=True))
        return dZ

    def update_parameters(self, optimization):
        pass

    def save_parameters(self, file_path):
        pass

    def load_parameters(self, file_path):
        pass


class Model:

    def __init__(self, name, iTrainable, loss, optimization=None):
        self.name = name
        self.iTrainable = iTrainable
        self.loss = loss
        self.optimization = optimization
        if self.loss == "square_dist":
            self.loss_forward = self.square_dist
            self.loss_backward = self.dSquare_dist
        elif self.loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.dCross_entropy
        elif self.loss == "categorical_cross_entropy":
            self.loss_forward = self.categorical_cross_entropy
            self.loss_backward = self.dCategorical_cross_entropy
        else:
            raise ValueError("none square dist or cross entropy")

    def __str__(self):

        s = self.name + "\n"

        s += "\tLoss function: " + self.loss + "\n"

        s += f"\t{self.iTrainable}\n"

        return s

    def forward_propagation(self, prev_A):
        return self.iTrainable.forward_propagation(prev_A)

    @staticmethod
    def to_categorical(y, num_classes=None):
        y = np.array(y, dtype="int")
        input_shape = y.shape

        if num_classes is None:
            num_classes = np.max(y) + 1

        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1

        return categorical

    def square_dist(self, Y_hat, Y):

        errors = (Y_hat - Y) ** 2
        return errors

    def dSquare_dist(self, Y_hat, Y):

        m = Y.shape[1]
        dY_hat = 2 * (Y_hat - Y) / m
        return dY_hat

    def compute_cost(self, Y_hat, Y):

        m = Y.shape[1]
        errors = self.loss_forward(Y_hat, Y)
        J = np.sum(errors)
        return J / m

    def backward_propagation(self, Y_hat, Y):

        dY_hat = self.loss_backward(Y_hat, Y)
        return self.iTrainable.backward_propagation(dY_hat)

    def cross_entropy(self, Y_hat, Y):
        # Clip predictions to prevent log(0) which is undefined
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # Binary classification case
        # if Y.ndim == 1 or Y.shape[1] == 1:
        #     loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

        # Multi-class classification case
        # else:
        # Reshape Y_hat to match the number of classifications

        # Calculate the loss
        loss = -np.sum(Y * np.log(Y_hat), axis=1)
        return np.mean(loss)

    def dCross_entropy(self, Y_hat, Y):
        # Clip predictions to prevent division by zero
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # Compute the gradient for binary classification
        if Y.ndim == 1 or Y.shape[1] == 1:
            gradient = (Y_hat - Y) / (Y_hat * (1 - Y_hat))

        # Compute the gradient for multi-class classification
        else:
            gradient = Y_hat - Y

        return gradient

    def categorical_cross_entropy(self, Y_hat, Y):
        # Clip predictions to prevent log(0) which is undefined
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)
        # Compute the cross-entropy loss for each sample
        loss = -np.sum(Y * np.log(Y_hat), axis=1)
        # Return the average loss over all samples
        return np.mean(loss)

    def dCategorical_cross_entropy(self, Y_hat, Y):
        # Clip predictions to prevent division by zero
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # Compute the gradient
        gradient = Y_hat - Y

        # Return the gradient
        return gradient

    def train(self, x_train, y_train, num_iterations, verbose=True):
        errors = []
        for i in range(num_iterations):
            error = 0
            for X, Y in zip(x_train, y_train):
                # forward propagation
                Y_hat = self.forward_propagation(X)

                # error
                error += self.loss_forward(Y_hat, Y)

                # backward propagation
                self.backward_propagation(Y_hat, Y)
                self.iTrainable.update_parameters(self.optimization)

            error /= len(x_train)
            errors.append(error)
            if verbose:
                print(
                    f"error after {i+1} updates ({((i+1)*100)//num_iterations}%): {error}"
                )
        return errors







