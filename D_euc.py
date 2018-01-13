#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE521 Assignment 1 - KNN Implementation for Classification and Regression

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

if __name__ == '__main__':
    
    sess = tf.InteractiveSession()
    
    X = tf.constant([[3, 4, 5], [5, 1, 1]])
    
    Z = tf.constant([[2, 4, 5], [1, 1, 1], [6,1,8]])
    
    print(sess.run(D_euc(X,Z)))