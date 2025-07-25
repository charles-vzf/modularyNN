import unittest
import os
import sys

# Add the parent directory to path to ensure imports work correctly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Import layers first
    from Layers import *
    LSTM_TEST = True
except BaseException as e:
    if str(e)[-6:] == "LSTM":
        LSTM_TEST = False
    else:
        raise e

# Import the optimization modules
from Optimization import *

# Local imports from the Helpers package
import Helpers
import Data
from Helpers.Helpers import gradient_check, gradient_check_weights, calculate_accuracy

# Import NeuralNetwork from local directory
from Helpers.NeuralNetwork import NeuralNetwork

# Standard library imports
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import argparse
import tabulate

# Import Data layer using relative path from project root
from Data import DatasetClasses

ID = 3  # identifier for dispatcher

class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_trainable(self):
        layer = FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable)
    
    def test_weights_size(self):
        layer = FullyConnected(self.input_size, self.output_size)
        self.assertTrue((layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)))

    def test_forward_size(self):
        layer = FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected(input_size, categories)
        init = TestFullyConnected.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.5))
        self.assertEqual(init.fan_in, input_size)
        self.assertEqual(init.fan_out, categories)
        if layer.weights.shape[0]>layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5)
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5)



class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size,:] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = ReLU()
        self.assertFalse(layer.trainable)

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU())
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestTanH(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = TanH()
        self.assertFalse(layer.trainable, msg="Error: trainable member for TanH is set incorrectly.")

    def test_forward(self):
        expected_tensor = 1 - 2 / (np.exp(2*self.input_tensor) + 1)

        layer = TanH()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = TanH()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1., msg="Error: Output of TanH should always be smaller than 1")
        self.assertGreaterEqual(out_min, -1., msg="Error: Output of TanH should always be greater than -1")

    def test_gradient(self):
        layers = list()
        layers.append(TanH())
        layers.append(L2Loss())
        difference = gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = Sigmoid()
        self.assertFalse(layer.trainable, msg="Error: trainable member for TanH is set incorrectly.")

    def test_forward(self):
        expected_tensor = 0.5 * (1. + np.tanh(self.input_tensor / 2.))

        layer = Sigmoid()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = Sigmoid()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1., msg="Error: Output of Sigmoid should always be smaller than 1")
        self.assertGreaterEqual(out_min, 0., msg="Error: Output of Sigmoid should always be greater than 0")

    def test_gradient(self):
        layers = list()
        layers.append(Sigmoid())
        layers.append(L2Loss())
        difference = gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax()
        self.assertFalse(layer.trainable)

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)))

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12)

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1
        input_tensor *= -10.
        layer = SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertAlmostEqual(element, 1/3, places = 3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places = 3)


    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)


    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax())
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T)


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(Loss.CrossEntropyLoss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0)

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4)


class TestOptimizers(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))

    def test_sgd_with_momentum(self):
        optimizer = Optimizers.SgdWithMomentum(1., 0.9)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]))

    def test_adam(self):
        optimizer = Optimizers.Adam(1., 0.01, 0.02)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, .5)
        np.testing.assert_almost_equal(result, np.array([-0.9814473195614205]))


class TestInitializers(unittest.TestCase):
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = []
            self.shape = (output_size, input_size)

        def initialize(self, initializer):
            self.weights = initializer.initialize(self.shape, self.shape[1], self.shape[0])

    def setUp(self):
        self.batch_size = 9
        self.input_size = 400
        self.output_size = 400
        self.num_kernels = 20
        self.num_channels = 20
        self.kernelsize_x = 41
        self.kernelsize_y = 41

    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return layer.shape, weights_after_init

    def test_uniform_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_uniform_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        p_value = stats.kstest(weights_after_init.flat, 'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_xavier_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_xavier_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        scale = np.sqrt(2) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_he_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.He())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.He())

        scale = np.sqrt(2) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)


class TestFlatten(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        self.input_tensor = self.input_tensor.reshape(self.batch_size, *self.input_shape)

    def test_trainable(self):
        layer = Flatten()
        self.assertFalse(layer.trainable)

    def test_flatten_forward(self):
        flatten = Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_vector = input_vector.reshape(self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9)

    def test_flatten_backward(self):
        flatten = Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9)


class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 105
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = Conv((1, 1), self.kernel_shape, self.num_kernels)
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        conv = Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))

    def test_forward_size_stride(self):
        conv = Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, 4, 7))

    def test_forward_size_stride_uneven_image(self):
        conv = Conv((3, 2), self.kernel_shape, self.num_kernels + 1)
        input_tensor = np.array(range(int(np.prod(self.uneven_input_shape) * (self.batch_size + 1))), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, ( self.batch_size+1, self.num_kernels+1, 4, 8))

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv((1, 1), (1, 3, 3), 1)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_multi_channel(self):
        np.random.seed(1337)
        maps_in = 2
        bias = 1
        conv = Conv((1, 1), (maps_in, 3, 3), 1)
        filter = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
        conv.bias = np.array([bias])
        input_tensor = np.random.random((1, maps_in, 10, 14))
        expected_output = bias
        for map_i in range(maps_in):
            expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv((1, 1), (3, 3, 3), 1)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3 , 10, 14))
        input_tensor[:,0] = tensor.copy()
        input_tensor[:,1] = tensor.copy()
        input_tensor[:,2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2)

    def test_1D_forward_size(self):
        conv = Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(3 * 15 * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape((self.batch_size, 3, 15))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,  (self.batch_size,self.num_kernels, 8))

    def test_backward_size(self):
        conv = Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_backward_size_stride(self):
        conv = Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_1D_backward_size(self):
        conv = Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(45 * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape((self.batch_size, 3, 15))
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, 3, 15))

    def test_1x1_convolution(self):
        conv = Conv((1, 1), (3, 1, 1), self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_layout_preservation(self):
        conv = Conv((1, 1), (3, 3, 3), 1)
        conv.initialize(self.TestInitializer(), Initializers.Constant(0.0))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(np.squeeze(output_tensor) - input_tensor[:,1,:,:])), 0.)

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 36])
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv((2, 2), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = gradient_check_weights(layers, input_tensor, label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_weights_init(self):
        # simply checks whether you have not initialized everything with zeros
        conv = Conv((1, 1), (100, 10, 10), 150)
        self.assertGreater(np.mean(np.abs(conv.weights)), 1e-3)

    def test_bias_init(self):
        conv = Conv((1, 1), (1, 1, 1), 150 * 100 * 10 * 10)
        self.assertGreater(np.mean(np.abs(conv.bias)), 1e-3)

    def test_gradient_stride(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 35])
        input_tensor = np.abs(np.random.random((2, 6, 5, 14)))
        layers = list()
        layers.append(Conv((1, 2), (6, 3, 3), 1))
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_update(self):
        input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))
        conv = Conv((3, 2), self.kernel_shape, self.num_kernels)
        conv.optimizer = Optimizers.Sgd(1)
        conv.initialize(Initializers.He(), Initializers.Constant(0.1))
        # conv.weights = np.random.rand(4, 3, 5, 8)
        # conv.bias = 0.1 * np.ones(4)
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_initialization(self):
        conv = Conv((1, 1), self.kernel_shape, self.num_kernels)
        init = TestConv.TestInitializer()
        conv.initialize(init, Initializers.Constant(0.1))
        self.assertEqual(init.fan_in, np.prod(self.kernel_shape))
        self.assertEqual(init.fan_out, np.prod(self.kernel_shape[1:]) * self.num_kernels)


class TestPooling(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))

        self.categories = 12
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten())
        self.layers.append(L2Loss())
        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_trainable(self):
        layer = Pooling((2, 2), (2, 2))
        self.assertFalse(layer.trainable)

    def test_shape(self):
        layer = Pooling((2, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = Pooling((2, 1), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = Pooling((3, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_gradient_stride(self):
        self.layers[0] = Pooling((2, 2), (2, 2))
        difference = gradient_check(self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        label_tensor = np.random.random((self.batch_size, 24))
        self.layers[0] = Pooling((2, 1), (2, 2))
        difference = gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):
        label_tensor = np.random.random((self.batch_size, 6))
        self.layers[0] = Pooling((3, 2), (2, 2))
        difference = gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_layout_preservation(self):
        pool = Pooling((1, 1), (1, 1))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[4]]], [[[13]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[ 5.,  7.],[13., 15.]]],[[[21., 23.],[29., 31.]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.delta = 0.1
        self.regularizer_strength = 1337
        self.shape = (4, 5)

    def test_L1(self):
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -2
        weights_tensor = regularizer.calculate_gradient(weights_tensor)

        expected = np.ones(self.shape) * self.regularizer_strength
        expected[1:3, 2:4] *= -1

        difference = np.sum(np.abs(weights_tensor - expected))
        self.assertLessEqual(difference, 1e-10, msg="The calculate_gradient method in the "
                                                    "regularizers is needed to compute the new update in the"
                                                    " optimizers. For the L1 norm, it should return the element-wise "
                                                    "sign of the weight tensor multiplied by the regularizer strength.")

    def test_L1_norm(self):
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -2
        norm = regularizer.norm(weights_tensor)
        self.assertAlmostEqual(norm, 24*self.regularizer_strength,
                               msg="Possible error: wrong computation. "
                                   "The norm method in the L1_Regularizer should return the sum of the absolute values"
                                   "of the tensor multiplied by the regularizer strength."
                               )

    def test_L2(self):
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor = regularizer.calculate_gradient(weights_tensor)

        difference = np.sum(np.abs(weights_tensor - np.ones(self.shape) * self.regularizer_strength))
        self.assertLessEqual(difference, 1e-10, msg="The calculate_gradient method in the "
                                                    "regularizers is needed to compute the new update in the"
                                                    " optimizers. For the L2 norm, it should return the weight tensor "
                                                    "multiplied by the regularizer strength.")

    def test_L2_norm(self):
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] += 1
        norm = regularizer.norm(weights_tensor)
        self.assertAlmostEqual(norm, 32 * self.regularizer_strength,
                               msg="Possible error: wrong computation. "
                                   "The norm method in the L2_Regularizer should return the L2 norm squared,"
                                   " i.e. sum of squared elements of the tensor, "
                                   "multiplied by the regularizer strength.")

    def test_L1_with_sgd(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Sgd(2)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -116, 2,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")

    def test_L2_with_sgd(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Sgd(2)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 268, 2,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")

    def test_L1_with_sgd_w_momentum(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.SgdWithMomentum(2,0.9)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -188, 1,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")

    def test_L2_with_sgd_w_momentum(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.SgdWithMomentum(2,0.9)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 196, 1,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")

    def test_L1_with_adam(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Adam(2, 0.9, 0.999)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -68, 2,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")

    def test_L2_with_adam(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Adam(2, 0.9, 0.999)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 188, 2,
                                       err_msg=
                                       "Make sure to have executed the following steps:\n"
                                       "- add a method 'add_regularizer' to the optimizer\n"
                                       "- add a member 'regularizer' to the optimizer that stores the regularizer\n"
                                       "- have all optimizers inherit from the 'Optimizer' class\n"
                                       "- use the 'calculate_gradient' method of the regularizers in the "
                                       "'calculate_update' method of each optimizer. E.g. you can compute right at the"
                                       " beginning the 'shrinked weights' and use those instead of the normal weights.")


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10
        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_trainable(self):
        layer = Dropout(0.25)
        self.assertFalse(layer.trainable, "Error: trainable flag is set to true.")

    def test_default_phase(self):
        drop_layer = Dropout(0.25)
        self.assertFalse(drop_layer.testing_phase, "Error: default for testing_phase should be false.")

    def test_forward_trainTime(self):
        drop_layer = Dropout(0.25)
        output = drop_layer.forward(self.input_tensor)
        self.assertEqual(np.max(output), 4)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean/self.input_size, 1., places=1,
                               msg="Make sure to implement inverted dropout:\n"
                                   "During training time:\n"
                                   "- randomly set activations to zero with probability '1-p'\n"
                                   "- multiply all activations by 1/p")

    def test_position_preservation(self):
        drop_layer = Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)
        error_prev = drop_layer.backward(self.input_tensor)
        np.testing.assert_almost_equal(np.where(output == 0.), np.where(error_prev == 0.),
                                       err_msg="Make sure to set the error tensor to zero in the"
                                               "positions where the input tensor was set to zero during"
                                               "the forward pass.")

    def test_forward_testTime(self):
        drop_layer = Dropout(0.5)
        drop_layer.testing_phase = True
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size,
                         msg="Make sure that during test time the input tensor is left unchanged"
                             "by the 'forward' method.")

    def test_backward(self):
        drop_layer = Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)
        self.assertEqual(np.max(output), 2, msg="Possible error: in the backward during training time"
                                            "all activations should be multiplied by 1/p")
        self.assertEqual(np.min(output), 0, msg="Possible error: in the backward the activations corresponding to those"
                                                "set to zero in the forward pass should also be set to zero.")

    def test_gradient(self):
        batch_size = 10
        input_size = 10
        input_tensor = np.ones((batch_size, input_size))
        label_tensor = np.zeros([batch_size, input_size])
        for i in range(batch_size):
            label_tensor[i, np.random.randint(0, input_size)] = 1
        layers = list()
        layers.append(Dropout(0.5))
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, label_tensor, seed=1337)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestBatchNorm(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 200
        self.channels = 2
        self.input_shape = (self.channels, 3, 3)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(0)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor_conv = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))

        self.categories = 5
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten())
        self.layers.append(FullyConnected(self.input_size, self.categories))
        self.layers.append(L2Loss())

        self.plot_shape = (self.input_shape[1], self.input_shape[0] * np.prod(self.input_shape[2:]))

    @staticmethod
    def _channel_moments(tensor, channels):

        tensor = np.transpose(tensor, (0, *range(2, tensor.ndim), 1))
        tensor = tensor.reshape(-1, channels)
        mean = np.mean(tensor, axis=0)
        var = np.var(tensor, axis=0)
        return mean, var

    def test_trainable(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        self.assertTrue(layer.trainable, "Error: trainable flag set to false.")

    def test_default_phase(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        self.assertFalse(layer.testing_phase, "Error: default phase set to true.")

    def test_forward_shape(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0], msg="Error: output shape is different from input.")
        self.assertEqual(output.shape[1], self.input_tensor.shape[1], msg="Error: output shape is different from input.")

    def test_forward_shape_convolutional(self):
        layer = BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor_conv)

        self.assertEqual(output.shape, self.input_tensor_conv.shape, msg="Error: output shape is different from input.")

    def test_forward(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0,
                               msg="Error: output does not have zero mean (along the channel dimension).")
        self.assertAlmostEqual(np.sum(np.square(var - np.ones(var.shape[0]))), 0,
                               msg="Error: output does not have unit variance (along the channel dimension.")

    def test_reformat_image2vec(self):
        layer = BatchNormalization(3)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reformat(image_tensor)
        np.testing.assert_equal(vec_tensor.shape, (120, 3))
        self.assertEqual(np.sum(vec_tensor, 1)[0], 72, msg="Error: wrong shape. In order to understand how"
                                                           " to implement the reformat function take a look at"
                                                           " the Regularization pdf. Notice that the reformat function "
                                                           "should take care of both cases: reshape an image to a vector"
                                                           " and viceversa.")
        self.assertEqual(np.sum(vec_tensor, 0)[0], 18660, msg="Error: wrong shape. In order to understand how"
                                                           " to implement the reformat function take a look at"
                                                           " the Regularization pdf. Notice that the reformat function "
                                                           "should take care of both cases: reshape an image to a vector"
                                                           " and viceversa.")

    def test_reformat_vec2image(self):
        layer = BatchNormalization(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape( 5, 3 , 6 , 4))
        vec_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(120, 3)
        image_tensor = layer.reformat(vec_tensor)
        np.testing.assert_equal(image_tensor.shape, (5, 3, 6, 4))
        self.assertEqual(np.sum(image_tensor, (0,1,2))[0], 15750, msg="Error: wrong shape. In order to understand how"
                                                           " to implement the reformat function take a look at"
                                                           " the Regularization pdf. Notice that the reformat function "
                                                           "should take care of both cases: reshape an image to a vector"
                                                           " and viceversa.")
        self.assertEqual(np.sum(image_tensor, (0,2,3))[0], 21420, msg="Error: wrong shape. In order to understand how"
                                                           " to implement the reformat function take a look at"
                                                           " the Regularization pdf. Notice that the reformat function "
                                                           "should take care of both cases: reshape an image to a vector"
                                                           " and viceversa.")

    def test_reformat(self):
        layer = BatchNormalization(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4))
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reformat(image_tensor)
        image_tensor2 = layer.reformat(vec_tensor)
        np.testing.assert_allclose(image_tensor, image_tensor2, err_msg="Error! Make sure that you can reshape a 4D "
                                                                        "image to a vector and compute again the image "
                                                                        "from the vector using the 'reformat' function.")

    def test_forward_convolutional(self):
        layer = BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor_conv)
        mean, var = TestBatchNorm._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0, msg="Make sure to use the reformat method in the forward pass"
                                                               "and to transform back the output before returning it.")
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0, msg="Make sure to use the reformat method in the forward pass"
                                                               "and to transform back the output before returning it.")

    def test_forward_train_phase(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0,
                            msg="Make sure to use the batch mean and batch variance in the"
                                " computation of x tilde.")

    def test_forward_train_phase_convolutional(self):
        layer = BatchNormalization(self.channels)
        layer.forward(self.input_tensor_conv)

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor_conv, self.channels)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0,
                            msg="Make sure to use the batch mean and batch variance in the"
                                " computation of x tilde.")

    def test_forward_test_phase(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)
        layer.testing_phase = True

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0,
                               msg="Make sure to use the test mean. The test mean is computed during training time"
                                   " as a moving average. It is then kept fixed during test time.")
        self.assertAlmostEqual(np.sum(np.square(var)), 0,
                               msg="Make sure to use the test variance. The test variance is computed during training time"
                                   " as a moving average. It is then kept fixed during test time."
                               )

    def test_forward_test_phase_convolutional(self):
        layer = BatchNormalization(self.channels)
        layer.forward(self.input_tensor_conv)
        layer.testing_phase = True

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor_conv, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0,
                               msg="1. Make sure that reformatting is working\n"
                                   "2. Make sure to use the test mean. The test mean is computed during training time"
                                   " as a moving average. It is then kept fixed during test time."
                               )
        self.assertAlmostEqual(np.sum(np.square(var)), 0,
                               msg="1. Make sure that reformatting is working\n"
                                   "2. Make sure to use the test variance. The test variance is computed during training time"
                                   " as a moving average. It is then kept fixed during test time."
                               )

    def test_gradient(self):
        self.layers[0] = BatchNormalization(self.input_tensor.shape[-1])
        difference = gradient_check(self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_gradient_weights(self):
        self.layers[0] = BatchNormalization(self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias(self):
        self.layers[0] = BatchNormalization(self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_convolutional(self):
        self.layers[0] = BatchNormalization(self.channels)
        difference = gradient_check(self.layers, self.input_tensor_conv, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = gradient_check_weights(self.layers, self.input_tensor_conv, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = gradient_check_weights(self.layers, self.input_tensor_conv, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_update(self):
        layer = BatchNormalization(self.input_tensor.shape[-1])
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros_like(self.input_tensor)
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))



class TestRNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 13
        self.output_size = 5
        self.hidden_size = 7
        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

        self.categories = 4
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_initialization(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        init = TestFullyConnected.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.0))
        if layer.weights.shape[0] > layer.weights.shape[1]:
            self.assertEqual(np.sum(layer.weights), 21.0, msg="Make sure to provide a property named 'weights' that allows"
                                                          " to access the weights of the first FC layer.")
        else:
            self.assertEqual(np.sum(layer.weights), 60.0, msg="Make sure to provide a property named 'weights' that allows"
                                                          " to access the weights of the first FC layer.")

    def test_trainable(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        self.assertTrue(layer.trainable, "Error: trainable flag set to false.")

    def test_forward_size(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size, msg="Possible error: wrong weights' shapes in "
                                                                       "one of the FC layers.")
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_forward_stateful(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)

        input_vector = np.random.rand(self.input_size, 1).T
        input_tensor = np.tile(input_vector, (2, 1))

        output_tensor = layer.forward(input_tensor)

        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0,
                            msg="Possible error: hidden state is not updated.")

    def test_forward_stateful_TBPTT(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        layer.memorize = True
        output_tensor_first = layer.forward(self.input_tensor)
        output_tensor_second = layer.forward(self.input_tensor)
        self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0,
                            msg="Possible error: hidden state is not updated.")

    def test_backward_size(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = RNN(self.input_size, self.hidden_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.output_size, self.batch_size]).T
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4,
                             msg="To compute the gradient we recommend using the 'backward' methods"
                                 "of each layer. In order to use the backward method, we need to restore the correct"
                                 " 'state' of each layer, that is we need to set the activations of that layer to the "
                                 "correct ones (at each time step t these activations were different). Once all "
                                 "activations are set correctly we can proceed and compute the gradient by going back"
                                 " through time: starting from time step T going back to time step T-1, T-2, ...,  0."
                                 "In this way we will compute the gradient at each time step. The result should be then"
                                 " reversed such that we obtain the gradient starting from time step 0.")

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-4, msg="If you have implemented the structure correctly, "
                                                           "the gradient weights will be computed automatically by the "
                                                           "FC layers when you call the 'backward' method. In doing so "
                                                           "we have a different value of the gradient_weights for each "
                                                           "time step. The gradient_weights used to compute the update "
                                                           "for the FC layers is the sum of all gradients at each step."
                                                           "Thus, you should accumulate the gradient_weights when going"
                                                           " back through time and use the sum to compute the update."
                                                           " Remember that you can access these gradient using the "
                                                           "property named 'gradient_weights'. ")

    def test_weights_shape(self):
        layer = RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        self.assertTrue(hasattr(layer, 'weights'), msg='your RNN layer does not provide a weights attribute')
        fc_layer = FullyConnected(20, 7)
        self.assertIn(layer.weights.shape, [fc_layer.weights.shape, fc_layer.weights.T.shape])

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = RNN(100000, 100, 1)
        layer.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0, "Possible error: bias in the FC layers is not implemented.")


if LSTM_TEST:
    class TestLSTM(unittest.TestCase):
        def setUp(self):
            self.batch_size = 9
            self.input_size = 13
            self.output_size = 5
            self.hidden_size = 7
            self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

            self.categories = 4
            self.label_tensor = np.zeros([self.categories, self.batch_size]).T
            for i in range(self.batch_size):
                self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        def test_trainable(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)
            self.assertTrue(layer.trainable)

        def test_forward_size(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)
            output_tensor = layer.forward(self.input_tensor)
            self.assertEqual(output_tensor.shape[1], self.output_size)
            self.assertEqual(output_tensor.shape[0], self.batch_size)

        def test_forward_stateful(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)

            input_vector = np.random.rand(1, self.input_size)
            input_tensor = np.tile(input_vector, (self.input_size, 1))

            output_tensor = layer.forward(input_tensor)

            self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)

        def test_weights_shape(self):
            layer = LSTM(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            self.assertTrue(hasattr(layer, 'weights'), msg='your LSTM layer does not provide a weights member')
            fc_layer = FullyConnected(20, 28)
            self.assertIn(layer.weights.shape, [fc_layer.weights.shape, fc_layer.weights.T.shape])

        def test_forward_stateful_TBPTT(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)
            layer.memorize = True
            output_tensor_first = layer.forward(self.input_tensor)
            output_tensor_second = layer.forward(self.input_tensor)
            self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0)

        def test_backward_size(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = layer.backward(output_tensor)
            self.assertEqual(error_tensor.shape[1], self.input_size)
            self.assertEqual(error_tensor.shape[0], self.batch_size)

        def test_update(self):
            layer = LSTM(self.input_size, self.hidden_size, self.output_size)
            layer.optimizer = Optimizers.Sgd(1)
            for _ in range(10):
                output_tensor = layer.forward(self.input_tensor)
                error_tensor = np.zeros([self.output_size, self.batch_size]).T
                error_tensor -= output_tensor
                layer.backward(error_tensor)
                new_output_tensor = layer.forward(self.input_tensor)
                self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

        def test_gradient(self):
            input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
            layers = list()
            layer = LSTM(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            layers.append(layer)
            layers.append(L2Loss())
            difference = gradient_check(layers, input_tensor, self.label_tensor)
            self.assertLessEqual(np.sum(difference), 1e-4)

        def test_gradient_weights(self):
            input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
            layers = list()
            layer = LSTM(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            layers.append(layer)
            layers.append(L2Loss())
            difference = gradient_check_weights(layers, input_tensor, self.label_tensor, False)
            self.assertLessEqual(np.sum(difference), 1e-3)

        def test_bias(self):
            input_tensor = np.zeros((1, 100000))
            layer = LSTM(100000, 100, 1)
            layer.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())
            result = layer.forward(input_tensor)
            self.assertGreater(np.sum(result), 0)


class TestNeuralNetwork3(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_append_layer(self):
        # this test checks if your network actually appends layers, whether it copies the optimizer to these layers, and
        # whether it handles the initialization of the layer's weights
        net = NeuralNetwork(Optimizers.Sgd(1),
                                          Initializers.Constant(0.123),
                                          Initializers.Constant(0.123))
        fcl_1 = FullyConnected(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2)
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer)
        self.assertTrue(np.all(net.layers[0].weights == 0.123))

    def test_data_access(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Sgd(1e-4),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2)

    def test_iris_data(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Sgd(1e-3),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(iterations = 4000, metrics_interval = 4000, plot_interval=None)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_regularization_loss(self):
        '''
        This test checks if the regularization loss is calculated for the fc and rnn layer and tracked in the
        NeuralNetwork class
        '''
        import random
        fcl = FullyConnected(4, 3)
        rnn = RNN(4, 4, 3)

        for layer in [fcl, rnn]:
            loss = []
            for reg in [False, True]:
                opt = Optimizers.Sgd(1e-3)
                if reg:
                    opt.add_regularizer(Constraints.L1_Regularizer(8e-2))
                net = NeuralNetwork(opt,Initializers.Constant(0.5),
                                                      Initializers.Constant(0.1))

                net.data_layer = Data.datasets.IrisData(100, random = False)
                net.loss_layer = Loss.CrossEntropyLoss()
                net.append_layer(layer)
                net.append_layer(SoftMax())
                net.train(1, metrics_interval=4000, plot_interval=None)
                loss.append(np.sum(net.loss))

            self.assertNotEqual(loss[0], loss[1], "Regularization Loss is not calculated and added to the overall loss "
                                                  "for " + layer.__class__.__name__)

    def test_iris_data_with_momentum(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.SgdWithMomentum(1e-3, 0.8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(2000, metrics_interval=4000, plot_interval=None)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Momentum.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Adam(1e-3, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(3000, metrics_interval=4000, plot_interval=None)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_ADAM.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization(input_size))
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        net.train(2000, metrics_interval=4000, plot_interval=None)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Batchnorm.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout(0.3))
        net.append_layer(SoftMax())

        net.train(2000, metrics_interval=4000, plot_interval=None)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Dropout.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_layer_phases(self):
        np.random.seed(None)
        net = NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Data.datasets.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization(input_size))
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout(0.3))
        net.append_layer(SoftMax())

        net.train(100, metrics_interval=4000, plot_interval=None)

        data, labels = net.data_layer.get_test_set()
        results = net.test(data)

        bn_phase = net.layers[0].testing_phase
        drop_phase = net.layers[4].testing_phase

        self.assertTrue(bn_phase)
        self.assertTrue(drop_phase)

    def test_digit_data(self):
        adam = Optimizers.Adam(5e-3, 0.98, 0.999)
        self._perform_test(adam, TestNeuralNetwork3.iterations, 'ADAM', False, False)

    def test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'L2_regularizer', False, False)

    def test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(sgd_with_l1, TestNeuralNetwork3.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Optimizers.Adam(1e-2, 0.98, 0.999)
        self._perform_test(adam, TestNeuralNetwork3.iterations, 'Batch_norm', False, True)

    def test_all(self):
        sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        np.random.seed(None)
        net = NeuralNetwork(optimizer,
                                          Initializers.He(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 150
        num_kernels = 4

        net.data_layer = Data.datasets.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        if batch_norm:
            net.append_layer(BatchNormalization(1))

        cl_1 = Conv(conv_stride_shape, convolution_shape, num_kernels)
        net.append_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_layer(BatchNormalization(num_kernels))

        net.append_layer(ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.append_layer(Flatten())

        fcl_1 = FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        if batch_norm:
            net.append_layer(BatchNormalization(fcl_1_input_size//2))

        if dropout:
            net.append_layer(Dropout(0.3))

        net.append_layer(ReLU())

        fcl_2 = FullyConnected(int(fcl_1_input_size / 2), int(fcl_1_input_size / 3))
        net.append_layer(fcl_2)

        net.append_layer(ReLU())

        fcl_3 = FullyConnected(int(fcl_1_input_size / 3), categories)
        net.append_layer(fcl_3)

        net.append_layer(SoftMax())

        net.train(iterations, metrics_interval=4000, plot_interval=None)
        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.), file=f)
        print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        self.assertGreater(accuracy, 0.3)



class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == "__main__":

    import sys
    if sys.argv[-1] == "Bonus":
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestTanH, TestSigmoid, TestConstraints, TestDropout, TestBatchNorm, TestRNN, TestNeuralNetwork3]
        percentages = [2.5, 2.5, 5, 5, 25, 40, 20]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 3
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p),
                          "{:.3f} / 10 (%)".format(p / 100 * exam_percentage)])
        table.append([])
        table.append(["Ex3", "Total Achieved", "", "{} / 100 (%)".format(total_points),
                      "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])

        print(tabulate.tabulate(table, headers=['Pos', 'Test', "Result", 'Percent', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()