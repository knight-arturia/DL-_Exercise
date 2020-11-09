from matplotlib import pyplot as plt
import numpy as np


class Checker():
    def  __init__(self,resolution,tile_size):
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
        a = np.concatenate((np.ones(self.til),np.zeros(self.til)))
        # padding half of the Checker in a line form, and reshape in a square form
        b = np.pad(a, int((self.Res**2) / 2 - self.til), 'wrap').reshape((self.Res, self.Res))
        # b plus b_T to get the entire checkerboard
        c = (b + b.T==1).astype(int)
        return c
    
    def show(self):
        checker = self.draw()
        plt.imshow(checker, cmap='gray')
        plt.show()
        

class Circle():
    def  __init__(self, resolution, radius, position):
        # integer resolution of entire graph
        self.Res = resolution
        # integer radius of the circle
        self.Rad = radius
        # tuple discrib center of circle
        self.x_0 = position[0]
        self.y_0 = position[1]
        
    def draw(self):
        square = np.ones((self.Res, self.Res))
        r = np.arange(0, self.Rad, 1)
        
        for i in r:
            # raw position of point in circle
            x = np.arange(self.x_0 - i, self.x_0 + i, 1)
            for j in x:
                # raw position of point in circle
                y = int(self.y_0 + np.sqrt(i**2 - (j - self.x_0)**2))
                square[j][y] = 0
        return square

    def show(self):
        Cir = self.draw()
        plt.imshow(Cir, cmap='gray')
        plt.show()