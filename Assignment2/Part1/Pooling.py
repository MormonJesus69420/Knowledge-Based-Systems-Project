from math import inf

from numpy import ndarray, zeros
from typing import Tuple
from enum import IntEnum


class PoolingType(IntEnum):
    MAX_POOLING = 0
    MIN_POOLING = 1

    def __str__(self) -> str:
        return self.name.title()


class Pooling:
    def __init__(self, pool_size: Tuple = (2, 2), pooling_type: PoolingType = PoolingType.MAX_POOLING):
        self._pool_size_x, self._pool_size_y = pool_size
        self._pooling_type = pooling_type

    def pool(self, feature_map: ndarray) -> ndarray:
        size_x = int(feature_map.shape[0] / self._pool_size_x)
        size_y = int(feature_map.shape[1] / self._pool_size_y)

        result = zeros((size_x, size_y))

        for y in range(size_y):
            start_y = y * self._pool_size_y
            end_y = start_y + self._pool_size_y

            for x in range(size_x):
                start_x = x * self._pool_size_x
                end_x = start_x + self._pool_size_x

                result[x, y] = self.get_pool(end_x, end_y, feature_map, start_x, start_y)

        return result

    def get_pool(self, end_x: int, end_y: int, feature_map: ndarray, start_x: int, start_y: int) -> float:
        if self._pooling_type is PoolingType.MAX_POOLING:
            return feature_map[start_x:end_x, start_y:end_y].max()
        elif self._pooling_type is PoolingType.MIN_POOLING:
            return feature_map[start_x:end_x, start_y:end_y].min()
        else:
            return -inf
