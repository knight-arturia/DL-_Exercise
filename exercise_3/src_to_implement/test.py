from matplotlib.pyplot import axis
import numpy as np

a = np.arange(21).reshape(21,)
print(a)

b = np.arange(7).reshape(7,)
print(b)

c = np.outer(a,b)

print(c[0:3])