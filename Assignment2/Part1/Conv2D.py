from numpy import pad, ndarray, array, zeros, multiply, sum
from enum import IntEnum
from typing import Tuple


def get_kernel_arr():
    edge_kernel = array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    identity_kernel = array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    return [edge_kernel, identity_kernel]


class KernelType(IntEnum):
    EDGE = 0
    IDENTITY = 1

    def __str__(self) -> str:
        return self.name.title()


class Convolution:
    def __init__(self, strides: Tuple = (1, 1), kernel_type: KernelType = KernelType.EDGE, debug: bool = False):
        self._kernel_x, self._kernel_y = (3, 3)
        self._stride_x, self._stride_y = strides
        self._kernels = get_kernel_arr()
        self._kernel = kernel_type
        self._debug = debug

    def apply(self, feature_map: ndarray):
        shape = feature_map.shape
        target = zeros(shape)
        feature_map = self.pad_feature_map(feature_map)
        kernel = self.get_kernel()

        if self._debug:
            print(f'Padded feature map:\n{feature_map}')
            print(f'Kernel type: {self._kernel}')
            print(f'Kernel:\n{kernel}')

        for y in range(target.shape[1]):
            start_y = y * self._stride_y
            end_y = start_y + self._kernel_y
            for x in range(target.shape[0]):
                start_x = x * self._stride_x
                end_x = start_x + self._kernel_x

                f_map = feature_map[start_x:end_x, start_y:end_y]

                target[x, y] = sum(multiply(f_map, kernel))

        return target

    def pad_feature_map(self, feature_map: ndarray, padding: Tuple = (1, 1)) -> ndarray:
        return pad(feature_map, padding, 'constant', constant_values=0)

    def get_kernel(self):
        return self._kernels[self._kernel]
