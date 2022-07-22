""" <Local Varying Anomaly Detection>
    Copyright (C) <2022>  <Wen-Yan Lin>
    daniellin@smu.edu.sg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import numpy as np
import random
from scipy.io import loadmat


def trainTestSplit_oneClass(allFeat, allGt, target, numSamples=100):
    mask = np.where(allGt ==target)[0]
    trainInd = random.sample(list(mask), numSamples)    
    trainFeat = np.copy(allFeat[trainInd])
    testFeat = allFeat.copy()
    testFeat = np.delete(testFeat, trainInd, axis =0)
    testGt = allGt.copy()
    testGt = np.delete(testGt, trainInd, axis =0)
    return trainFeat, testFeat, testGt

def trainTestSplit_multiClass(allFeat, allGt, numSamples=100):
    numClass = int(np.max(allGt)+1)
    trainFeat = []
    testFeat = allFeat.copy()
    testGt = allGt.copy()
    for i in range(numClass):
        trainF, testFeat, testGt = trainTestSplit_oneClass(testFeat, testGt, i, numSamples=numSamples)
        trainFeat.append(trainF)   
    return trainFeat, testFeat, testGt
        

def unstackFeat(feat_):
    feat = np.concatenate(feat_, axis=0)
    gt = np.zeros(feat.shape[0], dtype=int)
    label = 0
    cur = 0
    for f in feat_:
        gt[cur:cur+f.shape[0]] = label
        label = label + 1
        cur = cur + f.shape[0]
    return feat, gt



def importData(setIndex):

    folderNames = ['fashion-mnist',
                   'STL-10', 
                   'fake-stl10', 
                   'MIT-Places-Small',  
                   'dogvscat',
                   'mnist']
    base_ = '../data/resNet/'

    print('Dataset: ' + folderNames[setIndex])
    if setIndex == 0:
        import tensorflow as tf
        mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train

        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2


    elif setIndex == 1:
        allFeat = np.load(base_ + '/STL-10/resNet50.npy')
        gtAll = np.load(base_ + '/STL-10/gt.npy')
        trainFeat_, testFeat, testGt = trainTestSplit_multiClass(allFeat, gtAll, 1200)
        trainFeat, trainGt = unstackFeat(trainFeat_)

    elif setIndex == 2:
        testFeat = np.load(base_ + '/STL-10/resNet50.npy')
        testGt = np.load(base_ + '/STL-10/gt.npy')
        trainFeat = np.load(base_ + '/fake-stl10/resNet50.npy')
        trainGt = np.load(base_ + '/fake-stl10/gt.npy')

    elif setIndex == 3:
        allFeat = np.load(base_ + '/MIT-Places-Small/resNet50.npy')
        gtAll = np.load(base_ + '/MIT-Places-Small/gt.npy')
        # class 5 is arches, which overlaps with abbey, making evaluation results questionable
        mask = gtAll<5
        allFeat = allFeat[mask]
        gtAll = gtAll[mask]
        trainFeat_, testFeat, testGt = trainTestSplit_multiClass(allFeat, gtAll, 2000)
        trainFeat, trainGt = unstackFeat(trainFeat_)

    elif setIndex == 4:
        testFeat = np.load(base_ + 'dogvscat/cats_vs_dogs224feats/feats_test.npy')
        testGt = np.load(base_ + 'dogvscat/cats_vs_dogs224feats/y_test.npy')

        trainFeat = np.load(base_ + 'dogvscat/cats_vs_dogs224feats/feats_train.npy')
        trainGt = np.load(base_ + 'dogvscat/cats_vs_dogs224feats/y_train.npy')


    elif setIndex == 5:
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train
        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2

    elif setIndex == 6:
        train = loadmat('../Data/svnh_numbers/train_32x32.mat')
        x_train = train['X']
        y_train = np.squeeze(train['y']) -1

        test = loadmat('../Data/svnh_numbers/test_32x32.mat')
        x_test = test['X']
        y_test = np.squeeze(test['y']) -1


        x_train = np.einsum('ijkl->lijk', x_train)
        x_test = np.einsum('ijkl->lijk', x_test)


        x_train = x_train.reshape(x_train.shape[0], 32*32*3)
        x_test = x_test.reshape(x_test.shape[0], 32*32*3)



    else:
        print('Data-set ' + setIndex + 'is not defined.')


    return trainFeat, trainGt, testFeat, testGt, folderNames[setIndex]


def import_data_with_noise(setIndex, noise_feat, per=0.1):
    x_train, y_train, x_test, y_test, folder_name = importData(setIndex)

    num_class = np.max(y_train) + 1
    x_train_ = [x_train]
    y_train_ = [y_train]
    pos = range(noise_feat.shape[0])
    for i in range(num_class):
        mask = y_train==i
        num = int(per*sum(mask))
        ind = random.sample(pos, num)
        x_train_.append(noise_feat[ind])
        y_train_.append(i*np.ones(num, dtype=int))

    x_train_ = np.concatenate(x_train_, axis=0)
    y_train_ = np.concatenate(y_train_)

    return x_train_, y_train_, x_train, y_train,  x_test, y_test, folder_name
    