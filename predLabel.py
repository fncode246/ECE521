# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:20:16 2018

@author: Helena
"""


import numpy as np
import tensorflow as tf
from PIL import Image

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
            train_target: D dim vector, with targets for the training data points 
            test_data: N2 X D, with N2 test data points
    Output:
            test_target: N2 dim vector, with targets for the test data points
            
    '''
    # compute the distances between train and test data points
    distances = D_euc(test_data, train_data) 
    #-get the k nearest distances, -1*distances b/c tf.nn.top_k() finds the greatest k distances   
    nearest_k_train_values, nearest_k_indices = tf.nn.top_k(-1*distances, k=K)
    
    # for each test data point
    # use for-loop b/c tf.unique_with_counts() takes 1D input only
    target_shape = [test_data.shape[0]]
    test_targets = tf.zeros(target_shape,tf.int32)
    for i in range(test_data.shape[0]):
        # get the nearest k training targets for each test data point
        nearest_k_targets = tf.gather(train_target,nearest_k_indices[i,:])  
        # tally the targets of the nearest k training data
        targets, idx, counts = tf.unique_with_counts(tf.reshape(nearest_k_targets,shape=[-1]))
        # find the target with the higheset occurence in nearest_k_targets
        max_count, max_count_idx=tf.nn.top_k(counts, k=1)
        # the most frequent occuring target to the output target vector
        test_target = tf.gather(targets,max_count_idx)
        sparse_test_target = tf.SparseTensor([[i,]], test_target, target_shape)
        test_targets = tf.add(test_targets, tf.sparse_tensor_to_dense(sparse_test_target))        
    return test_targets
    
def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget     

if __name__ == '__main__':
    # set up data
    test_mode = 1       # 0 for facial recognition, 1 for gender, 2 for a toy test case
    if test_mode == 2:  # toy data test case, with D=2
        trainData = [[1,1],[2,2],[1,3],[3,3],[4,4],[4,2]]
        trainTarget = [1, 1, 1, 0, 0, 0]
        validData = [[1,2],[4,3],[1,2],[4,1]]
        validTarget=[1, 0, 1, 0]
        testData = [[2,1]]
        testTarget = [1]
    else: 
        trainData, validData, testData, trainTarget, validTarget, testTarget \
            = data_segmentation('./data.npy', './target.npy', test_mode)

    # build graph
    # cannot use a placeholder because of the line test_targets = tf.zeros(target_shape,tf.int32)
    train_data = tf.Variable(trainData, dtype=tf.float32)
    train_target = tf.Variable(trainTarget, dtype=tf.int32)
    valid_data = tf.Variable(validData, dtype=tf.float32)
    valid_target = tf.Variable(validTarget, dtype=tf.int32)
    test_data = tf.Variable(testData, dtype=tf.float32)
    test_target = tf.Variable(testTarget, dtype=tf.int32)
    
    # start session 
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    if test_mode == 2: 
        K = 3
        valid_estimate =  predLabel(valid_data, train_data, train_target, K)        
        # loss function: count the total # of misclassifications
        loss = tf.count_nonzero(tf.not_equal(valid_estimate, valid_target))
        print("\nFor k = {}, there are {} misclassifications".format(K,sess.run(loss)))
    else:
        # run validation to select K 
        K_list = (1, 5, 10, 25, 50, 100, 200)
        for K in K_list: 
            valid_estimate =  predLabel(valid_data, train_data, train_target, K) 
            loss = tf.count_nonzero(tf.not_equal(valid_estimate, valid_target))
            print("\nFor k = {}, there are {} misclassifications".format(K,sess.run(loss)))
              
        # test with the k selected from validation 
        K_best=1
        test_estimate =  predLabel(test_data, train_data, train_target, K_best)  
        loss = tf.count_nonzero(tf.not_equal(test_estimate, test_target))
        print("\nFor k = {}, there are {} misclassifications".format(K_best,sess.run(loss)))
                    
        # analysis of misclassifications for K=10
        test_estimate =  predLabel(test_data, train_data, train_target, 10) 
        # get the first instance of misclassification 
        mis_idx = tf.where(tf.not_equal(test_estimate, test_target))[0]
        # get the 10 nearest neighbor training data of this failed test case 
        distances = D_euc(tf.gather(test_data,mis_idx), train_data) 
        nearest_k_train_values, nearest_k_indices = tf.nn.top_k(-1*distances, k=10)
        # save the failed test data and its neighbor training data as images 
        img = Image.fromarray(255*sess.run(tf.reshape(tf.gather(test_data,mis_idx),\
                                                      [32, 32]))).convert('LA')
        img.save('failed_case.png')
        for j in range(10):
            img = Image.fromarray(255*sess.run(tf.reshape(tf.gather(\
                 train_data,nearest_k_indices[:,j]), [32, 32]))).convert('LA')
            img.save('failed_case_neighbour'+str(j+1)+'.png')
       
