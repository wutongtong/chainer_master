#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2016-03-24

from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import data
import net

#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume',    '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu',       '-g', default=-1,   type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch',     '-e', default=20,   type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit',      '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', default=100,  type=int,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch   = args.epoch
n_units   = args.unit

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

#------------------------------------------------------------------------------
# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

x_train = x_train.reshape(N, 1, 28, 28)
x_test  = x_test.reshape(N_test, 1, 28, 28)

#------------------------------------------------------------------------------
# Setup model
model = net.MnistNet()
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

#------------------------------------------------------------------------------
# Learning loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # Training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Forward and backward
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

    # Evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]), volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]), volatile='on')

        # Forward
        loss = model(x, t)

        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

#------------------------------------------------------------------------------
# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
