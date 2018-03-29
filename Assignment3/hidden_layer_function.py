#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:58:06 2018

ECE521 Assignment 3 - Neural Network Hidden Layer Function with 
Xavier Initializaton

@author: Krist Papadopoulos
"""

import numpy as np
import tensorflow as tf

#Create the dataset as per the assignment code
with np.load('notMNIST.npz') as data:
    Data, Target = data ["images"], data["labels"]  
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    
#reshape input data to be a training point by 784 dimensions
trainData_rs = trainData.reshape(trainData.shape[0],-1)
validData_rs = validData.reshape(validData.shape[0],-1)
testData_rs =  testData.reshape(testData.shape[0],-1)
    
#create tensor for training data    
X_train_input = tf.constant(trainData_rs, dtype=tf.float32)  

    
#function to compute the sum of the inputs times the weights of the nodes in hidden layer
def hidden_layer(X, hidden_units):
    x_dimension = X.shape[1].value
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    hidden_weights = tf.Variable(initializer([x_dimension, hidden_units]), name='weights')
    hidden_biases = tf.Variable(tf.zeros(hidden_units), name='biases')
    
    return tf.add(tf.matmul(X, hidden_weights), hidden_biases)

#create 1 hidden layer with 1000 nodes
h1 = hidden_layer(X_train_input,1000)
#create another layer adding relu and 10 nodes
h1 = hidden_layer(tf.nn.relu(h1),10)

init = tf.global_variables_initializer()

# test the hidden_layer function
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(h1)
    print(output)