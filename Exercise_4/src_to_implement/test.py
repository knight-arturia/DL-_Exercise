import torch
import numpy as np
from sklearn.metrics import f1_score

# a = np.array([])
# # a.append(torch.Tensor([1, 2]))
# # a.append(torch.Tensor([0, 1]))
# a = np.append(a, np.array([1,2]))
# a = np.append(a, np.array([0,1]))
# print(a)

# b = np.array([])
# # b.append(torch.Tensor([0, 2]))
# # b.append(torch.Tensor([1, 1]))
# b = np.append(b, np.array([1,2]))
# b = np.append(b, np.array([2,1]))
# print(b)

# c = f1_score(a, b, average='micro')
# print(c)

res = []
train_losses = np.array([1,2,3,4])
vali_losses = np.array([1,2])
res.append(train_losses)
res.append(vali_losses)
print(res)