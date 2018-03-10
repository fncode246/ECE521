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
with np.load("notMNIST.npz") as data:
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
    
    # convert to one hot
    trainZeros=np.zeros((15000, 10))
    trainZeros[np.arange(15000),trainTarget]=1
    trainTarget = trainZeros
    validZeros=np.zeros((1000, 10))
    validZeros[np.arange(1000),validTarget]=1
    validTarget = validZeros
    testZeros=np.zeros((2724, 10))
    testZeros[np.arange(2724),testTarget]=1
    testTarget = testZeros
    
#Define parameters used in the model
learning_rates = [0.001]
regularization = 0.01
iterations = 5000
batchSize = 500

total_batches = int(len(trainData_rs)/batchSize)
epochs = int(iterations/total_batches)

LR_1_1 = tf.Graph()
with LR_1_1.as_default():  
    #training inputs
    X_train_input = tf.constant(trainData_rs, dtype=tf.float32)        
    y_train_input = tf.constant(trainTarget, dtype=tf.float32)
    X, y = tf.train.slice_input_producer([X_train_input, y_train_input], num_epochs=None)          
    X_batch, y_batch = tf.train.batch([X, y], batch_size=batchSize)
    
    # validation and test
    X_val = tf.constant(validData_rs, dtype=tf.float32)        
    y_val = tf.constant(validTarget, dtype=tf.float32)
    X_test = tf.constant(testData_rs, dtype=tf.float32)        
    y_test = tf.constant(testTarget, dtype=tf.float32)
    
    w = tf.Variable(tf.random_normal([784, 10]), name="w")        
    b = tf.Variable(tf.zeros([1]), name="b")       
    y_pred = tf.matmul(X_batch, w) + b

    # setup the optimization problem
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")   
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=y_pred)) \
                + 0.5*regularization * tf.nn.l2_loss(w)
    
########################3############# 2.1.1 #######################################
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  
        
    # Initialize the curves used in the plot
    loss_array_train = np.zeros([len(learning_rates),epochs+1])      
    loss_array_val = np.zeros([len(learning_rates),epochs+1])  
    acc_array_train = np.zeros([len(learning_rates),epochs+1])      
    acc_array_val = np.zeros([len(learning_rates),epochs+1]) 
    for idx, i in enumerate(learning_rates):
        print('Learning Rate: {:2}'.format(i))   # display progress      
        startTime = time.time()
        with tf.Session() as sess:
            #initializie the session and global variables
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer(),
            ])
            # batch training using queues
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
                 
            iter_counter = 0                
            for j in range(iterations):                
                _, loss_value = sess.run([optimizer,loss], feed_dict={learning_rate: i})
                iter_counter += 1                   
                duration = time.time() - startTime 
                # For every integer epoch, calculate the losses and accuracies for plotting          
                if iter_counter % total_batches == 0:                       
                    #display progress
                    print('Epoch: {:4}, Loss: {:5f}, Duration: {:2f}'. \
                          format(int(iter_counter/total_batches), loss_value, duration))                  
                    y_pred = tf.matmul(X_batch, w) + b 
                    y_val_pred = tf.matmul(X_val, w) + b
                    # training loss                      
                    loss_array_train[idx, int(iter_counter/total_batches)] = loss_value
                    # validation loss
                    loss_array_val[idx, int(iter_counter/total_batches)] = sess.run( \
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits\
                                       (labels=y_val, logits=y_val_pred)) \
                        + 0.5*regularization * tf.nn.l2_loss(w))
                    # training accuracy
                    acc_array_train[idx, int(iter_counter/total_batches)] = \
                        sess.run(tf.count_nonzero(tf.equal((tf.argmax(y_pred,1)), tf.argmax(y_batch,1))))
                    # validation accuracy
                    acc_array_val[idx, int(iter_counter/total_batches)] = \
                        sess.run(tf.count_nonzero(tf.equal((tf.argmax(y_val_pred,1)), tf.argmax(y_val,1))))
            
            # test accuracy
            y_test_pred = tf.matmul(X_test, w) + b
            test_acc = tf.count_nonzero(tf.equal(tf.argmax(y_test_pred,1), tf.argmax(y_test,1)))
            print("% test accuracy is: ")
            print(sess.run(test_acc)/2724*100)
                                    
            coord.request_stop()
            coord.join(threads)
            
    # Plots
    plt.figure(figsize=(10,10))
    plt.title('Losses for training and validation')      
    plt.scatter(np.arange(epochs), loss_array_train[0,1:epochs+1], marker='x', color='r', label = 'training')
    plt.scatter(np.arange(epochs), loss_array_val[0,1:epochs+1], marker='d', color='b', label = 'validation')
    plt.legend(loc='upper right')        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.title('Accuracies for training and validation')
    plt.scatter(np.arange(epochs), acc_array_train[0,1:epochs+1]/batchSize*100, marker='o', color='r', label = 'training')
    plt.scatter(np.arange(epochs), acc_array_val[0,1:epochs+1]/1000*100, marker='.', color='g', label = 'validation')
    plt.legend(loc='upper right')        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()