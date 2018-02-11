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
from sklearn.metrics import mean_squared_error


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


def k_nearest_neighbours(X_train, X_test, k):
    
    '''
    Computes the k nearest neighbours using tensorflow based on the 
    squared euclidean dist to the test point (X_test) to the training points (X_train)
    
    Input:  X_train tensor of shape (N X 1), X_test tensor of shape (M X 1), tf variable k # of nearest neighbours
    Output: k_nearest_indices, tensor of shape (M X k)
    '''
    
    neg_dist = tf.negative(dist_euclid(X_test, X_train))
    k_nearest_values, k_nearest_indices = tf.nn.top_k(neg_dist, k=k, sorted=True, name='k_nearest')
    return k_nearest_indices


def responsibility_matrix(X_train, X_test, nearest_indices, k):
    
    '''
    Computes the responsibility matrix using numpy based on the 
    indices of the nearest neighbours to each test point
    
    Input:  X_train array of shape (N X 1), X_test array of shape (M X 1), nearest_indices array of shape (M x k)
            k # of nearest neighbours
    Output: Resonsibility Matrix, rm, array of shape (M X N)
    
    '''
    
    rm = np.zeros([X_test.shape[0],X_train.shape[0]])
    rm[np.arange(X_test.shape[0])[:, np.newaxis],nearest_indices] = 1/k
    return rm


def knn_predict(y_train, r_matrix):
    '''
    The regression predictions are produced by taking taking responsibility multipled by the trainTarget values
    
    Input: y (trainTarget) array of shape N X 1, responsibility matrix (r_matrix) array of shape (M X N)
    Output: predictions, y_pred, array of shape (M X 1)
    '''
    
    y_pred = y_train.transpose().dot(r_matrix.transpose()).reshape(-1,1)
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
        k =         tf.placeholder(dtype=tf.int32, name='k')
    
        with tf.name_scope('predictions'):
            X_test = tf.placeholder(dtype=tf.float32, name='X_test')
            k_nearest_indices = k_nearest_neighbours(X, X_test, k)
    
        init_op = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            for K in k_list:
                print("\nFor k = {}:".format(K))
                
                for data, target, name in zip(data_set, target_set, set_names):
                    #for each k, print the squared loss of the train, validate and test dataset
                    print("The {} set MSE is:".format(name)) 
                    #compute the k-nearest neighbour indices using tensorflow to return an array of indices
                    indices = sess.run(k_nearest_indices, feed_dict={X: trainData, X_test: data, k: K})
                    #compute the responsibility matrix for the test points and k nearest neighbours
                    r_m = responsibility_matrix(trainData, data, indices, K)
                    #print MSE between targets and knn predictions
                    print(mean_squared_error(target, knn_predict(trainTarget,r_m)))
                    
                X_Data = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]
                
                #compute the nearest neighbours to the new data X_Data using tensorflow function
                X_Data_n_n = sess.run(k_nearest_indices, feed_dict={X: trainData, X_test: X_Data, k: K})
                #compute the responsibility matrix for the test points to the trainData
                X_Data_rm = responsibility_matrix(trainData, X_Data, X_Data_n_n, K)
                #compute the predictions of the X_Data targets using trainTarget and the responsibility matrix
                X_Data_pred = knn_predict(trainTarget, X_Data_rm)
                
                plt.title('k-NN Regression Predictions on data1D, k={}'.format(K))
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.scatter(x=Data, y=Target)
                plt.step(x=X_Data, y=X_Data_pred, c='r')
                plt.show()
                    
