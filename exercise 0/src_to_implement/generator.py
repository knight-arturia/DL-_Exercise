import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random
import cv2


class ImageGenerator:
    def __init__(self,
                 file_path,
                 label_path,
                 batch_size,
                 image_size,
                 rotation=False,
                 mirroring=False,
                 shuffle=False):

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        # import the labels of images
        # The labels are stored in json format and can be directly loaded as
        # dictionary.
        with open(label_path, 'r') as load_f:
            self.image_labels = json.load(load_f)

        #for next()
        self.image_list = list(self.image_labels.keys())
        if self.shuffle:
            random.shuffle(self.image_list)
        else:
            self.image_list.sort()

        self.pointer = 0

        #TODO: implement constructor

    """
    This function creates a batch of images and corresponding labels and returns them.
    In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
    Note that your amount of total data might not be divisible without remainder with the batch_size.
    Think about how to handle such cases
    """

    def next(self):
        # This function creates a batch of images and corresponding labels and
        # returns them.
        # In this context a "batch" of images just means a bunch, say 10 images
        # that are forwarded at once.
        # Note that your amount of total data might not be divisible without
        # remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        #Which images need to be imported?
        num_images = self.batch_size
        images_2_import = []

        while num_images > 0:

            print(num_images, self.pointer, len(self.image_list))
            if self.pointer + num_images > len(self.image_list):

                images_2_import = images_2_import + self.image_list[self.pointer:]
                num_images = num_images - (len(self.image_list) - self.pointer)
                self.pointer = 0

            else:

                images_2_import = images_2_import + self.image_list[
                    self.pointer:(self.pointer + num_images)]
                self.pointer = self.pointer + num_images
                num_images = 0

        #import images
        ist_das_erst = True

        for image_index in images_2_import:

            file_name = self.file_path + image_index + '.npy'

            an_image = np.load(file_name)

            #resize

            an_image = cv2.resize(an_image,
                                  (self.image_size[0], self.image_size[1]),
                                  interpolation=cv2.INTER_AREA)

            #flags

            #rotation
            if self.rotation:

                an_image = np.rot90(an_image,
                                    k=random.randint(1, 3),
                                    axes=(0, 1))

            #mirroring

            if self.mirroring:

                #Vertical mirror
                if random.choice([True, False]):

                    an_image = an_image[::-1]

                #Horizontal mirror
                if random.choice([True, False]):

                    an_image = an_image[::-1]
                    an_image = np.rot90(an_image, k=2, axes=(0, 1))

            #shuffle <- did in __init__

            an_image = np.expand_dims(an_image, 0)

            if ist_das_erst:

                images = an_image
                labels = np.array([int(self.image_labels[image_index])],
                                  dtype=np.int8)

                ist_das_erst = False

            else:

                images = np.append(images, an_image, axis=0)

                labels = np.append(
                    labels,
                    np.array([int(self.image_labels[image_index])],
                             dtype=np.int8))

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random
        # transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        #rotation
        if random.choice([True, False]):
            img = np.rot90(img, k=random.randint(1, 3), axes=(0, 1))

        #mirroring
        if random.choice([True, False]):

            #Vertical mirror
            if random.choice([True, False]):
                img = img[::-1]

            #Horizontal mirror
            if random.choice([True, False]):
                img = img[::-1]
                img = np.rot90(img, k=2, axes=(0, 1))

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        return self.class_dict[x]

    def show(self):

        images, show_labels = self.next()

        fig, axs = plt.subplots(3, 4, figsize=(16, 11))
        fig.subplots_adjust(hspace=.5, wspace=0.001)
        axs = axs.ravel()

        for i in range(12):

            axs[i].axis('off')

            if i < self.batch_size:
                image = images[i, :, :, :]
                show_label = self.class_name(show_labels[i])
                axs[i].set_title(show_label)
                axs[i].imshow(image)

        plt.show()

        return
        # In order to verify that the generator creates batches as required,
        # this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
