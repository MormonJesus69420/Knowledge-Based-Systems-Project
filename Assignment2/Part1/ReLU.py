from numpy import vectorize, ndarray


class ReLU:
    def relu(self, x: float) -> float:
        return max(x, 0.0)

    def activate(self, feature_map: ndarray) -> ndarray:
        vector = vectorize(self.relu)
        return vector(feature_map)
