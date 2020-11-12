from matplotlib import pyplot as plt
import numpy as np


class Checker():
    def __init__(self, resolution, tile_size):
        # number of pixel an each dimension
        self.Res = resolution
        # number of pixel an individual tile
        self.til = tile_size
        pass

    def draw(self):
        if (self.Res % (2 * self.til)):
            print('ERROR: resolution must dividable by 2*tile_size')
            return False
        # get the first part in the first pixel line
        a = np.concatenate((np.ones(self.til), np.zeros(self.til)))
        # padding half of the Checker in a line form, and reshape in a square form
        b = np.pad(a, int((self.Res**2) / 2 - self.til), 'wrap').reshape(
            (self.Res, self.Res))
        # b plus b_T to get the entire checkerboard
        c = (b + b.T == 1).astype(int)
        return c

    def show(self):
        checker = self.draw()
        plt.imshow(checker, cmap='gray')
        plt.show()


class Circle():
    def __init__(self, resolution, radius, position):
        # integer resolution of entire graph
        self.Res = resolution
        # integer radius of the circle
        self.Rad = radius
        # tuple discrib center of circle
        self.x_0 = position[0]
        self.y_0 = position[1]

    def draw(self):
        square = np.zeros((self.Res, self.Res))
        r = self.Rad
        # raw position of point in circle
        x = np.arange(self.x_0 - r, self.x_0 + r, 1)
        # raw position of point in circle
        y = np.arange(self.y_0 - r, self.y_0 + r, 1)
        for i in x:
            for j in y:
                if (i - self.x_0)**2 + (j - self.y_0)**2 <= r**2:
                    #continue
                    square[i][j] = 1
                else:
                    continue
        return square

    def show(self):
        Cir = self.draw()
        plt.imshow(Cir, cmap='gray')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        """
        docstring
        """
        pass
    def draw(parameter_list):
        """
        docstring
        """
        pass
