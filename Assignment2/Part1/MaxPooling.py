from numpy import ndarray, zeros
from typing import Tuple


class Pooling:
    def __init__(self, pool_size: Tuple = (2, 2)):
        self._pool_size_x, self._pool_size_y = pool_size

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

                result[x, y] = feature_map[start_x:end_x, start_y:end_y].max()

        return result
