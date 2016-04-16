#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2016-03-24

import chainer
import chainer.functions as F
import chainer.links as L

"""
Model definition
"""
class MnistNet(chainer.Chain):

    def __init__(self):
        """
        Trainable and primitive structure with parameter variable.
        - Convolution2D(in_channels, out_channels, kernel_size)
        - Linear(in_size, out_size)
        """
        super(MnistNet, self).__init__(
            conv1 = L.Convolution2D(1,32,5,pad=2),
            conv2 = L.Convolution2D(32,64,5,pad=2),
            fc3   = L.Linear(7*7*64,1024),
            fc4   = L.Linear(1024,10)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()

        """
        Functions on variable with backpropagations ability.
        - relu(x)
        - max_pooling_2d(x, kernel_size)
        - dropout(x, ratio=0.5, train=True)
        """
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc3(h))
        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.fc4(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
