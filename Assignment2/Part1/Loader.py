import os as os
from genericpath import isfile
from os import listdir
from os.path import join

from numpy import array, dot, ndarray
from matplotlib.image import imread


class Loader:
    def __init__(self, path: str = "Letters/", relative: bool = True):
        if relative:
            dir = os.path.dirname(__file__)
            path = dir + '/' + path

        self.path = path
        self.letters = [f for f in listdir(path) if isfile(join(path, f))]

        self._train_x = self._test_x = self._train_y = self._test_y = None

        self.load_training_set()

    def load_training_set(self) -> None:
        train_x, test_x, train_y, test_y = ([], [], [], [])

        for letter in self.letters:
            arr = self.rgb2gay(imread(self.path + letter)) / 255.0
            cat = self.get_category(letter)

            if self.check_validation_set(letter):
                test_x.append(arr)
                test_y.append(cat)
            else:
                train_x.append(arr)
                train_y.append(cat)

        print(train_x[0])

        self._train_x = array(train_x)
        self._test_x = array(test_x)
        self._train_y = array(train_y)
        self._test_y = array(test_y)

    def check_validation_set(self, filename: str) -> bool:
        return '_val' in filename

    def get_category(self, filename: str) -> int:
        return 0 if 'A' in filename else 1 if 'K' in filename else 2

    def rgb2gay(self, rgb: ndarray) -> ndarray:
        return dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @property
    def train_x(self) -> ndarray:
        return self._train_x

    @train_x.setter
    def train_x(self, val: ndarray):
        self._train_x = val

    @property
    def train_y(self) -> ndarray:
        return self._train_y

    @train_y.setter
    def train_y(self, val: ndarray):
        self._train_y = val

    @property
    def test_x(self) -> ndarray:
        return self._test_x

    @test_x.setter
    def test_x(self, val: ndarray):
        self._test_x = val

    @property
    def test_y(self) -> ndarray:
        return self._test_y

    @test_y.setter
    def test_y(self, val: ndarray):
        self._test_y = val
