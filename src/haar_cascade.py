#! /usr/bin/python3

import numpy as np
import cv2
import argparse

class HaarCascade:

    def __init__(self):
        self.dataset = []
        self.haar_matrix = []

        self.label = [chr(i + ord('A')) for i in range(25)]

    def train(self):
        for data in self.dataset:
            data = np.resize(data[:-1], (28, 28))
            tt = np.zeros(data.shape, dtype=int)

            for l in range(data.shape[0]):
                st = 1
                for r in range(data.shape[1]):
                    st += data[l, r] + tt[l - 1, r] + tt[l, r - 1]
                    tt[l, r] = st
            self.haar_matrix.append(tt) 
        print(self.dataset[0], self.haar_matrix[0])
            


    def load_dataset(self):

        with open('dataset.csv', 'r') as f:
            line = f.readline()
            while line:
                data = np.array(list(map(int, line.split(';'))))
                self.dataset.append(data)
                line = f.readline()
        


def main():
    haar = HaarCascade()
    haar.load_dataset()
    haar.train()

if __name__ == "__main__":
    main()
