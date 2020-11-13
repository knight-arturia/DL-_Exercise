from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
import json
import os

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


class ImageGenerator:


    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

                     
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',7: 'horse', 8: 'ship', 9: 'truck'}

        # import the labels of images
        # The labels are stored in json format and can be directly loaded as
        # dictionary.
        with open(label_path,'r') as load_f:
            self.image_labels = json.load(load_f)       

        #import images
        ist_das_erst = True
        for file_name in os.listdir(self.file_path):
            
            an_image = np.load(self.file_path + file_name)
            an_image = np.expand_dims(an_image,0)

            if ist_das_erst:
                self.dataset = an_image
                ist_das_erst = False
            else:
                self.dataset = np.append(self.dataset,an_image,axis=0)

        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and
        # returns them.
        # In this context a "batch" of images just means a bunch, say 10 images
        # that are forwarded at once.
        # Note that your amount of total data might not be divisible without
        # remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random
        # transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return
    def show(self):

        return
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method