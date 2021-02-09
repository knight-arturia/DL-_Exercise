import torch
import numpy as np
from sklearn.metrics import f1_score

a = []
a.append(np.array([1]))
print(a)

b = []
b.append(torch.Tensor([1]))
print(b)

c = f1_score(a, b)
print(c)