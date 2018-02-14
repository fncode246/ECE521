# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:44:56 2018

@author: Helena
"""


import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#Create the dataset as per the assignment code
with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]
    
    #reshape input data to be a training point by 784 dimensions
    trainData_rs = trainData.reshape(trainData.shape[0],-1)
    validData_rs = validData.reshape(validData.shape[0],-1)
    testData_rs =  testData.reshape(testData.shape[0],-1)
    
#Define parameters used in the model
adamOpt = 1
learning_rates = [0.001]
regularization = 0.01
iterations = 5000
batchSize = 500

total_batches = int(len(trainData_rs)/batchSize)
epochs = int(iterations/total_batches)

LR_1_1 = tf.Graph()
with LR_1_1.as_default():  
    with tf.name_scope("queue-inputs"):
        #training inputs
        X_train_input = tf.constant(trainData_rs, dtype=tf.float32)        
        y_train_input = tf.constant(trainTarget, dtype=tf.float32)
        X, y = tf.train.slice_input_producer([X_train_input, y_train_input], num_epochs=None)          
        X_batch, y_batch = tf.train.batch([X, y], batch_size=batchSize)
        
    with tf.name_scope("model"):        
        w = tf.Variable(tf.zeros([784, 1]), name="w")        
        b = tf.Variable(tf.zeros([1]), name="b")       
        y_pred = tf.sigmoid(tf.matmul(X_batch, w) + b)
        
    with tf.name_scope("hyperparameters"):
        learning_rate = tf.placeholder(tf.float32, name="learning-rate")
    
    with tf.name_scope("loss-function"):   
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred, logits=y_batch)) \
                    + 0.5*regularization * tf.nn.l2_loss(w)
        
    with tf.name_scope("train"):  
        if adamOpt: 
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else: 
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
    loss_array = np.zeros([len(learning_rates),epochs+1])           
    for idx, i in enumerate(learning_rates):
        print('Learning Rate: {:2}'.format(i))        
        startTime = time.time()
        with tf.Session() as sess:
            #initializie the session and global variables
            sess.run(tf.global_variables_initializer())            
            #start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                 
            iter_counter = 0                
            for j in range(iterations):                
                _, loss_value = sess.run([optimizer,loss], feed_dict={learning_rate: i})
                iter_counter += 1                   
                duration = time.time() - startTime            
                if j == 0:
                    loss_array[idx,j] = 0                    
                elif iter_counter % total_batches == 0:                       
                    #print status to stdout.
                    print('Epoch: {:4}, Loss: {:5f}, Duration: {:2f}'. \
                          format(int(iter_counter/total_batches), loss_value, duration))                        
                    loss_array[idx, int(iter_counter/total_batches)] = loss_value
                
            coord.request_stop()
            coord.join(threads)