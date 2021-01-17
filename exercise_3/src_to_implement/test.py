from matplotlib.pyplot import axis
import numpy as np

weights = np.arange(0, 9).reshape(3,3)
print(weights)
out1 = np.linalg.norm(weights, ord=2, axis=None, keepdims=False)**2
out2 = np.linalg.norm(weights, ord=2, axis=None, keepdims=False)
a = None
if not a:
    print('test')
print(out1)
print(out2)