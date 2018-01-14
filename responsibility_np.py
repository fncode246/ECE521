#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE521 Assignment 1 - KNN Implementation for Classification and Regression

Implementation of the Responsibility function in Numpy

Created on Sun Jan 14 00:05:47 2018

@author: KP
"""
import numpy as np

def responsibility(test_point, train_data, k):
        
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
    
    r = np.zeros(train_data.shape[0])
    
    distances = D_euc(test_point.reshape(1,-1),train_data)
    
    nearest_neighbour_idx = np.argsort(distances).flatten()
    
    k_neighours_idx = nearest_neighbour_idx[:k]
    
    r[k_neighours_idx] = 1/k
    
    return r