
import numpy as np
import matplotlib.pyplot as plt
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator


#Checker
Checkerboard = Checker(600,60)
Checkerboard.show()

#Circle
Circlegraph = Circle(600, 100, (300,200))
Circlegraph.show()

#Spectrum
RGB_spec = Spectrum(600)
RGB_spec.show()

#ImageGenerator
label_path = './Labels.json'
file_path = './exercise_data/'
gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
gen.show()



                
        