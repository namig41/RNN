#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def show_digit(x, y):
    print("It is {}".format(y))
    plt.imshow(x.reshape(28, 28))
    plt.show()

def init_dataset(name):
    X_train = []
    Y_train = []

    with open(name, 'r') as f:
        csvfile = csv.reader(f, delimiter=',', quotechar='|')

        for row in csvfile:
            t = list(map(int, row))
            X_train.append(t[1:])
            Y_train.append(t[0])

    return np.array(X_train), np.array(Y_train)

def show_inf(l, a):
    plt.plot(l)
    plt.plot(a)
    plt.show()

def train(model, X_train, Y_train, n):
    BS = 32
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters()) 
    losses, accuracies = [], []

    for i in tqdm(range(n)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        optim.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        accuracies.append(accuracy)
        losses.append(loss)
        loss.backward()
        optim.step()

    show_inf(accuracies, losses)

class Net(torch.nn.Module):

    """Docstring for BoNet. """

    def __init__(self):
        """TODO: to be defined. """
        super(Net, self).__init__()
        
        self.l1 = nn.Linear(28*28, 128)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

def main():
    model = Net()
    X_train, Y_train = init_dataset('../data/mnist_dataset/mnist_train_100.csv')
    #train(model, X_train, Y_train, 100)

if __name__ == "__main__":
    main()
