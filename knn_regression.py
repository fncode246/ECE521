#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE521 Assignment 1 - KNN Implementation for Classification and Regression

KNN Regression Questions 1 and 2

Edited on Mon Jan 15, 2018

@author: Krist Papadopoulos, Logan Rooks, Yu Liu
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def dist_euclid(X, Z):
    
    '''
    Compute squared Euclidean L2 distance between tensor X and tensor Z
         
    Input: X is an N1 X D tensor, Z is is a N2 X D tensor 
    Output: ||X-Z||2/2 distance is a N1 X N2 tensor containing pairwise squared Euclidean distances
    
    X = tf.constant([[3, 4, 5], [5, 1, 1]])
    
    Z = tf.constant([[2, 4, 5], [1, 1, 1], [6,1,8]])
    
    dist_euclid(X,Z) = [[ 1 29 27]
                [34 16 50]]
    '''
    
    X_norm = tf.reshape(tf.reduce_sum(tf.square(X),axis=1), [-1,1])
    Z_norm = tf.reshape(tf.reduce_sum(tf.square(Z), axis=1), [1,-1])
    dist = X_norm + Z_norm - 2*tf.matmul(X,tf.transpose(Z))
    return dist


def k_nearest_neighbours(X, X_test, k):
    
    '''
    Computes the k nearest neighbours based on the squared euclidean dist to the test point (X_test) to the training points (X)
    '''
    
    neg_dist = tf.negative(dist_euclid(X_test, X))
    k_nearest_values, k_nearest_indices = tf.nn.top_k(neg_dist, k=k, sorted=True, name='k_nearest')
    return k_nearest_values, k_nearest_indices


def responsibility_matrix(X, X_test, k):
    
    neg_dist = tf.negative(dist_euclid(X_test, X))
    k_nearest_values, k_nearest_indices = tf.nn.top_k(neg_dist, k=k, sorted=True, name='k_nearest')
    r_mat_shape = (X.shape[0], X_test.shape[0])
    sparse_r_mat = tf.SparseTensor(indices=k_nearest_indices, values=1/k, dense_shape=r_mat_shape)
    r_mat = tf.add(tf.zeros(shape=r_mat_shape), sparse_r_mat, name="r_mat")
    return r_mat


def knn_predict(X, y, X_pred, k):
    
    #r_mat = responsibility_matrix(X, X_pred, k)
    _, k_nearest_indices = k_nearest_neighbours(X, X_pred, k)
    k_nearest_y = tf.gather(y, indices=k_nearest_indices, name='k_nearest_y')
    y_pred = tf.reduce_mean(k_nearest_y, axis=1)
    #y_pred = tf.matmul(tf.transpose(r_mat), y)
    return y_pred
   
if __name__ == '__main__':
    
    #Data generation given in the assignment
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
    
    data_set = (trainData, validData, testData)
    target_set = (trainTarget, validTarget, testTarget)
    set_names = ("train", "validation", "test")
    k_list = (1, 3, 5, 50)
    
    #Create Graph and Placeholders
    knn = tf.Graph()
    
    with knn.as_default():    
        X =         tf.placeholder(dtype=tf.float32, name='X')
        y =         tf.placeholder(dtype=tf.float32, name='y')
        k =         tf.placeholder(dtype=tf.int32, name='k')
        y_true =    tf.placeholder(dtype=tf.float32, name='y_true')
    
        with tf.name_scope('predictions'):
            X_pred = tf.placeholder(dtype=tf.float32, name='X_pred')
            y_pred = knn_predict(X, y, X_pred, k)
    
        with tf.name_scope('mse_loss'):
            loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    
        init_op = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            for K in k_list:
                print("\nFor k = {}:".format(K))
                
                for data, target, name in zip(data_set, target_set, set_names):
                    #for each k, print the squared loss of the train, validate and test dataset
                    print("For the {} set:".format(name))
                    error = sess.run(loss, feed_dict={X: trainData, y: trainTarget, X_pred: data, y_true: target, k: K})
                    print(error)
            
                X_Data = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]
                Pred = (sess.run(y_pred, feed_dict={X: trainData, y: trainTarget, X_pred: X_Data, k: K}))
                
                plt.title('KNN Regression Predictions on data1D, k={}'.format(K))
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.scatter(x=Data, y=Target)
                plt.step(x=X_Data, y=Pred, c='r')
                plt.show()
