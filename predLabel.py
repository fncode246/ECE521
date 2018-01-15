# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:20:16 2018

@author: Helena
"""


import numpy as np
import tensorflow as tf

def D_euc(X, Z):
    
    '''
    Code from Krist:
        
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


def predLabel(test_data, train_data, train_target, K):
        
    '''
    Input:
            K: the parameter k for KNN
            train_data: N1 X D, with N1 training data points, and each point 
                contains D features
            train_target: 1 X D, with targets for the training data points 
            test_data: N2 X D, with N2 test data points
    Output:
            test_target: N2 X 1, with targets for the test data points
            
    '''
    # compute the distances between train and test data points
    distances = D_euc(test_data, train_data) 
    #-get the k nearest distances, -1*distances b/c tf.nn.top_k() finds the greatest k distances   
    nearest_k_train_values, nearest_k_indices = tf.nn.top_k(-1*distances, k=K)
    
    # for each test data point
    for i in range(sess.run(tf.shape(test_data)[0])):
        # get the nearest k training targets for each test data point
        nearest_k_targets = tf.gather(train_target,nearest_k_indices[i,:])  
        # tally the targets of the nearest k training data
        targets, idx, counts = tf.unique_with_counts(tf.reshape(nearest_k_targets,shape=[-1]))
        # find the target with the higheset occurence in nearest_k_targets
        max_count, max_count_idx=tf.nn.top_k(counts, k=1)
        # append the most frequent occuring target to the output target vector
        if i==0:
            test_target = tf.gather(targets,max_count_idx)
        else: 
            test_target = tf.concat([test_target, tf.gather(targets,max_count_idx)],0)
    return tf.expand_dims(test_target,1)
     

if __name__ == '__main__':
    # toy data test case, with D=2
    trainData = [[1,1],[2,2],[1,3],[3,3],[4,4],[4,2]]
    trainTarget = [[1],[1],[1],[0],[0],[0]]
    validData = [[2,4],[4,3],[1,2],[4,1]]
    validTarget=[[1],[0],[1],[0]]

    train_data = tf.Variable(trainData, dtype=tf.float32)
    train_target = tf.Variable(trainTarget, dtype=tf.int32)
    valid_data = tf.Variable(validData, dtype=tf.float32)
    valid_target = tf.Variable(validTarget, dtype=tf.int32)
    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    K = 3
    valid_estimate =  predLabel(valid_data, train_data, train_target, K)
    print(sess.run(valid_estimate))



        