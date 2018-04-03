# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:55:21 2018

@author: Helena
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Get Data
with np.load('./notMNIST.npz') as data :
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
# one-hot 
trainZeros=np.zeros((15000, 10))
trainZeros[np.arange(15000),trainTarget]=1
trainTarget = trainZeros
validZeros=np.zeros((1000, 10))
validZeros[np.arange(1000),validTarget]=1
validTarget = validZeros
testZeros=np.zeros((2724, 10))
testZeros[np.arange(2724),testTarget]=1
testTarget = testZeros

# Extract batch_size batches randomly 
def grab_batches(trainData, trainTarget, batch_size):
    batch_indices = np.random.permutation(range(15000)).reshape(-1, batch_size)
    X_batches = trainData.reshape(-1, n_dim)[batch_indices]
    y_batches = trainTarget[batch_indices]
    batches = zip(X_batches, y_batches)
    return batches
    
# Hyperparam
learning_rate = 0.005
n_epochs = 800
batch_size = 500
weight_decays=[0]
drop = 1
    
# Setup training 
n_dim = 28*28
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,10])
# layer 1
initializer = tf.contrib.layers.xavier_initializer(uniform=False)
W1 = tf.Variable(initializer([X.shape[1].value, 1000]), name='weights')
b1 = tf.Variable(tf.zeros(1000), name='biases') 
S1 = tf.add(tf.matmul(X, W1), b1)
# layer 2
X2 = tf.nn.relu(S1)
initializer = tf.contrib.layers.xavier_initializer(uniform=False)
W2 = tf.Variable(initializer([X2.shape[1].value, 10]), name='weights')
b2 = tf.Variable(tf.zeros(10), name='biases')
if drop:
    X_drop = tf.nn.dropout(X2, keep_prob=0.5) 
else:
    X_drop = X2
y_ = tf.add(tf.matmul(X_drop, W2), b2)

loss_drop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
#regularizer = tf.nn.l2_loss(W)
#loss = tf.reduce_mean(loss + weight_decay * regularizer)

prediction = tf.cast(tf.round(tf.argmax(y_,1)), tf.int8)
equality = tf.equal(prediction, tf.cast(tf.argmax(Y,1), tf.int8))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

training_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_drop)
init = tf.global_variables_initializer()
    
# Training
valid_accuracies = []
train_accuracies = []
#test_accuracies = []
with tf.Session() as sess:
    for wd in weight_decays:
        sess.run(init)
        print("Weight Decay: {} \n".format(wd))
        for epoch in range(1,n_epochs+1):
            batches = grab_batches(trainData, trainTarget, batch_size)
            for X_batch, y_batch in batches:
                sess.run(training_step, feed_dict={X: X_batch, Y: y_batch})
            # Evaluate losses (without dropout)
            feed_dict ={X: trainData.reshape(-1,n_dim), Y: trainTarget}
            train_accuracy = sess.run(accuracy, feed_dict)
            print("Epoch: {}, Accuracy: {}".format(epoch, train_accuracy))
            train_accuracies.append(train_accuracy)
            valid_accuracy = sess.run(accuracy,feed_dict = {X: validData.reshape(-1,n_dim), Y: validTarget})
            valid_accuracies.append(valid_accuracy)
#            test_accuracy = sess.run(accuracy, feed_dict = {X: testData.reshape(-1,n_dim), Y: testTarget})
#            test_accuracies.append(test_accuracy)
        
    # Plots
    plt.figure(figsize=(10,10))
    plt.title('Accuracies for training and validation')
    plt.scatter(np.arange(n_epochs), train_accuracies, marker='x', color='r', label = 'training')
    plt.scatter(np.arange(n_epochs), valid_accuracies, marker='d', color='b', label = 'validation')
    #plt.scatter(np.arange(n_epochs), test_accuracies, marker='o', color='g', label = 'testing')
    plt.legend(loc='upper right')        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    # Visualize weights
    for ii in range(0,1000):
        weight_array=tf.reshape(W1[:,ii],[28,28])
        scale=255/sess.run(tf.reduce_max(tf.abs(weight_array)))
        img = Image.fromarray(scale*sess.run(weight_array)).convert('L')
        img.save('D50%\\Unit'+str(ii+1)+'.png')
#        arr = np.asarray(img)
#        plt.imshow(arr, cmap='gray')
    #print(sess.run(W1))
    
    
    
    
    