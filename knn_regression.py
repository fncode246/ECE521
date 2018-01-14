#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE521 Assignment 1 - KNN Implementation for Classification and Regression

KNN Regression Questions 1 and 2

Created on Fri Jan 12 11:37:10 2018

@author: KP
"""

import numpy as np
import tensorflow as tf

def D_euc(X, Z):
    
    '''
    Compute squared Euclidean L2 distance between tensor X and tensor Z
         
    Input: X is an N1 X D tensor, Z is is a N2 X D tensor 
    Output: ||X-Z||2/2 distance is a N1 X N2 tensor containing pairwise squared Euclidean distances
    
    X = tf.constant([[3, 4, 5], [5, 1, 1]])
    
    Z = tf.constant([[2, 4, 5], [1, 1, 1], [6,1,8]])
    
    D_euc(X,Z) = [[ 1 29 27]
                [34 16 50]]
    
    '''
    X_norm = tf.reshape(tf.reduce_sum(X**2,axis=1), [-1,1])
    Z_norm = tf.reshape(tf.reduce_sum(Z**2, axis=1), [1,-1])
    distance = X_norm + Z_norm - 2*tf.matmul(X,tf.transpose(Z))
    return distance


def responsibility(test_point, train_data, K):
        
    '''
    Compute the responsibilities of training points to a new test point (assigns k nearest training points to
    test point to value of 1/k, the rest to 0)
         
    Input: test_point 1 X D vector, training data is N X D vector, k is number of nearest neighbours
    Output: responsbility vector N X D
    
    test_point_value (for example) = 3 by 1 dimension
    
    train_data.shape = 80 points by 1 dimension
    
    nearest_neighbour_indices sorted by closest training points to test point = 
      [31, 60, 47, 66, 49, 48, 55, 27, 42, 76,  3, 52, 20, 40, 44, 78, 34,
       18, 13,  4, 26, 33,  1, 29, 17, 25, 22, 67, 54, 62, 23,  5, 32, 51,
       28, 39, 75, 43, 24, 61, 59, 63, 71,  2, 50, 69, 21,  7, 36, 45, 15,
       19, 53, 56, 65, 64, 72, 41, 11, 38, 74, 70, 68, 46,  0, 79,  8, 16,
       77, 30, 35, 10,  9, 58, 73, 57, 14,  6, 37, 12]
        
    r (for e.g. k = 3)[nearest_neighbour_idx = 31,60,47] = 
      [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.33333333,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.33333333,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.33333333,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
    
    '''
    #Tensorflow implementation of responsibility function - this needs testing and debugging
    r = tf.Variable(tf.zeros([train_data.shape[0],1])
    distances = D_euc(test_point, train_data)
    nearest_k_train_values, nearest_k_indices = tf.nn.top_k(distances, k=K, sorted = False)
    r_k = r[nearest_k_indices].assign(1/K)
    return r_k

if __name__ == '__main__':
    
    #data generation given in the assignment
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
    
    # Tensor Placeholders for train, validate and test data
    train_data =    tf.placeholder(shape=[None, 1], dtype=tf.float32)
    train_target =  tf.placeholder(shape=[None, 1], dtype=tf.float32)
    valid_data =    tf.placeholder(shape=[None, 1], dtype=tf.float32)
    valid_target =  tf.placeholder(shape=[None, 1], dtype=tf.float32)
    test_data =     tf.placeholder(shape=[None, 1], dtype=tf.float32)
    test_target =   tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    #Create the Tensorflow Graph
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # Code for predictions
    
 
    
    