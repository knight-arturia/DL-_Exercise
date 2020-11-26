from matplotlib.pyplot import axis
import numpy as np

# input_tensor = np.random.randint(-5, 5, size=10)
# print(input_tensor)
# neg_input = (input_tensor <= 0)
# output = input_tensor.copy()
# output[neg_input] = 0
# print(output)
a = np.array([[1,2],[3,4]])
b = np.array([5,6]).reshape(1,2)
c = np.r_[a,b]
print(c)

