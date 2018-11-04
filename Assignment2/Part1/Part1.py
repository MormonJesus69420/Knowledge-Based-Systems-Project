from Ogre import Layer
from Loader import Loader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    loader = Loader()

    layer1 = Layer(debug=True)
    layer2 = Layer(debug=True)

    for letter in loader.test_x:
        print("Layer 1:")
        letter = layer1.process(letter)
        print("Layer 2:")
        letter = layer2.process(letter)

        plt.figure()
        plt.imshow(letter, cmap=plt.get_cmap('nipy_spectral'))
        plt.colorbar()
        plt.show()
        plt.close()
