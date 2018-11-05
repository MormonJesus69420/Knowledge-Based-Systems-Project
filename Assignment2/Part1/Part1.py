from Ogre import Layer
from Loader import Loader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    loader = Loader()

    layer1 = Layer(debug=True)
    layer2 = Layer(debug=True)

    for letter in loader.test_x:

        plt.figure()
        plt.imshow(letter, cmap=plt.get_cmap('nipy_spectral'))
        plt.title('Before processing')
        plt.colorbar()
        plt.show()
        plt.close()

        print("Layer 1:")
        letter = layer1.process(letter)

        plt.figure()
        plt.imshow(letter, cmap=plt.get_cmap('nipy_spectral'))
        plt.title('After first layer')
        plt.colorbar()
        plt.show()
        plt.close()

        print("Layer 2:")
        letter = layer2.process(letter)

        plt.figure()
        plt.imshow(letter, cmap=plt.get_cmap('nipy_spectral'))
        plt.title('After second layer')
        plt.colorbar()
        plt.show()
        plt.close()
