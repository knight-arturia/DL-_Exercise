import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
dataset = pd.read_csv('data.csv')

trainset, testset = train_test_split(dataset, test_size=0.25, random_state=0)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl = t.utils.data.DataLoader(trainset)
test_dl = ChallengeDataset(testset, 'test')
print(train_dl)
print(test_dl)

# create an instance of our ResNet model
net = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.CrossEntropyLoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(net, criterion, optimizer, train_dl, test_dl, cuda= True, early_stopping_patience= 10)

# go, go, go... call fit on trainer
res = trainer.fit(20)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')