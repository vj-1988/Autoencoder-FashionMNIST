#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 21:54:44 2018

@author: vijay
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf

# =============================================================================
# prepare data
# =============================================================================

def gen_dat_labels(path):
    
    raw = pd.read_csv(path).values
    labels = raw[:,0]
    data = raw[:,1:]/256.0
    
    return labels,data
    


# =============================================================================
# main
# =============================================================================

def main():
    
    ## get labels and data from raw csv
    
    TrainLabels, TrainData = gen_dat_labels('fashionmnist/fashion-mnist_train.csv')
    TestLabels, TestData = gen_dat_labels('fashionmnist/fashion-mnist_test.csv')
    
    print (np.max(TrainData), np.min(TrainData))
    
    ## Network
    
    IpLayer = tf.placeholder(tf.float32, (None, 28, 28, 1), name='IpLayer')
    OpLayer = tf.placeholder(tf.float32, (None, 28, 28, 1), name='OpLayer')
    
    #convlayers
    
    C1 = tf.layers.conv2d(inputs=IpLayer, filters=128, kernel_size=(3,3),
                          padding='same', activation=tf.nn.relu)
    
    MP1 = tf.layers.max_pooling2d(C1, pool_size=(2,2), strides=(2,2),
                                  padding='same')
    
    C2 = tf.layers.conv2d(inputs=MP1, filters=128, kernel_size=(3,3),
                          padding='same', activation=tf.nn.relu)
    
    MP2 = tf.layers.max_pooling2d(C2, pool_size=(2,2), strides=(2,2),
                                  padding='same')
    
    C3 = tf.layers.conv2d(inputs=MP2, filters=128, kernel_size=(3,3),
                          padding='same', activation=tf.nn.relu)
    
    US1 = tf.image.resize_images(C3, size=(14,14),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    C4 = tf.layers.conv2d(inputs=US1, filters=128, kernel_size=(3,3),
                          padding='same', activation=tf.nn.relu)
    
    US2 = tf.image.resize_images(C4, size=(28,28),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    C5 = tf.layers.conv2d(inputs=US2, filters=128, kernel_size=(3,3),
                          padding='same', activation=tf.nn.relu)
    
    Flatten = tf.layers.conv2d(inputs=C5, filters=1, kernel_size=(3,3),
                               padding='same', activation=None)
    
    Reconstructed = tf.nn.sigmoid(Flatten)
    
    ## Optimizer
    loss = tf.losses.mean_pairwise_squared_error(labels=OpLayer, predictions=Reconstructed)
    
    cost = tf.reduce_mean(loss)
    opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


    ## start tensorflow session
    
    sess = tf.Session()
    epochs = 100
    batch_size = 100
    
    ## start training
    
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        
        for MiniBatch in range(TrainData.shape[0]//batch_size):
            
            TrainImgs = TrainData[((MiniBatch*batch_size)):((MiniBatch*batch_size)+batch_size)].reshape((-1, 28, 28, 1))
            
            batch_cost, _ = sess.run([cost, opt], feed_dict={IpLayer: TrainImgs,
                                                         OpLayer: TrainImgs})

    
            print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

if __name__=='__main__':
    main()