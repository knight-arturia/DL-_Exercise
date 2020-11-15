from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape


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
        # padding half of the Checker in a line form, and reshape in a square
        # form
        b = np.pad(a, int((self.Res ** 2) / 2 - self.til), 'wrap').reshape((self.Res, self.Res))
        # b plus b_T to get the entire checkerboard
        self.output = (b + b.T == 1).astype(int)

        return self.output

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
       # x = np.arange(self.x_0 - r, self.x_0 + r, 1)
        # raw position of point in circle
       # y = np.arange(self.y_0 - r, self.y_0 + r, 1)
        for i in range(self.Res):
            for j in range(self.Res):
                if ((j - self.x_0) ** 2 + (i - self.y_0) ** 2) <= (self.Rad ** 2):#(j - self.x_0)**2 + (i - self.y_0)**2 <= r**2:
                    #continue
                    square[i][j] = 1

        
        self.output = square

        return self.output

    def show(self):
        Cir = self.draw()
        plt.imshow(Cir, cmap='gray')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        self.Res = resolution
    
    def draw(self):
        if self.Res > 4072:
            print('Resolution is too large.\n')
            return 0
        # multiple the given
        a = np.outer(np.arange(0, 256), np.ones(256, dtype=np.uint8))
        b = np.sqrt(np.outer(np.arange(255, -1, -1), np.arange(255, -1, -1, dtype = np.uint8)))
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[:,:,0] = a.T
        rgb[:,:,1] = a
        rgb[:,:,2] = b

        self.output = rgb

        return rgb

    def show(self):
        Spec = self.draw()
        plt.imshow(Spec)
        plt.show()


