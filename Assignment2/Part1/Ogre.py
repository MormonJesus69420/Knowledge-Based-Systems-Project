from numpy import ndarray

from Conv2D import Convolution, KernelType
from Pooling import Pooling
from ReLU import ReLU


class Layer:
    def __init__(self, debug: bool = False):
        self._convolution = Convolution(kernel_type=KernelType.EDGE, debug=debug)
        self._relu = ReLU()
        self._pooling = Pooling()
        self._debug = debug

    def process(self, feature_map: ndarray) -> ndarray:
        if self._debug:
            print(f'Initial feature map:\n{feature_map}')

        result = self._convolution.apply(feature_map)

        if self._debug:
            print(f'Feature map after convolution:\n{result}')

        result = self._relu.activate(result)

        if self._debug:
            print(f'Feature map after ReLU:\n{result}')

        result = self._pooling.pool(result)

        if self._debug:
            print(f'Feature map after max pooling:\n{result}')

        return result
