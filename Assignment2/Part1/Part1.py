import numpy as np

from MaxPooling import Pooling
from Ogre import Layer


if __name__ == "__main__":
    layer1 = Layer(debug=True)
    layer2 = Layer(debug=True)

    print("Layer 1:")
    arr = np.array([[15, 9, 4, -12], [11, -1, 8, 6], [8, 7, -55, 17], [2, -72, 7, 3]])
    arr = layer1.process(arr)
    print("Layer 2:")
    layer2.process(arr)

