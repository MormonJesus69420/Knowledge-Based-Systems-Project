from numpy import vectorize, ndarray


class ReLU:
    def activate(self, feature_map: ndarray) -> ndarray:
        return vectorize(lambda x: max(x, 0.0))(feature_map)
