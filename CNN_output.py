from __future__ import print_function
from __future__ import division
from TestMatFiles import X_Test, X_Train, Y_Train, Y_Test
import numpy as np
import tensorflow as tf

mini_batch = 150
Learning_rate = 0.5
Steps = 1000

def Convolution(X_As_Image, size):
    W = weightsMatrix(size)
    b = biasMatrix([size[3]])
    Just_Convolution = tf.nn.conv2d(X_As_Image, W, strides=[1,1,1,1], padding='SAME')
    After_Relu = tf.nn.relu(Just_Convolution + b)  # add biases
    return After_Relu

def Subsampling(Image, h):
    Subsampled_Image = tf.nn.max_pool(Image, ksize=[1, h, h, 1], strides=[1, 2, 2, 1], padding='SAME')
    return Subsampled_Image

def weightsMatrix(size):
    w = tf.Variable(tf.truncated_normal(size, stddev=0.09), dtype=tf.float32)
    print(size)
    return w

def biasMatrix(size):
    b = tf.Variable(tf.zeros(size), dtype=tf.float32)
    print(size)
    return b

def propagate(X_Train):
	g_first = tf.Graph()
	with g_first.as_default():
		tf_X_Train = tf.placeholder(tf.float32, shape=(None, X_Train.shape[1]))
    	
    	X_Train_AS_Image = tf.reshape(tf_X_Train, [-1, 28, 28, 1])
    	
    	First_Convolution = Convolution(X_Train_AS_Image, size=[5, 5, 1, 32])
    	Subsampled_First_Convolution = Subsampling(First_Convolution, h=2)
    	
    	Second_Convolution = Convolution(Subsampled_First_Convolution, size=[5, 5, 32, 64])
    	Subsampled_Second_Convolution = Subsampling(Second_Convolution, h=2 )
    	
    	Flattened_Everything = tf.reshape(Subsampled_Second_Convolution, [ -1, 7*7*64])  # Batch_size X 3136
    	No_of_Nodes_In_First_Flat_Layer = int(Flattened_Everything.shape[1])
    	
    	w1 = weightsMatrix([No_of_Nodes_In_First_Flat_Layer, 2048])
    	b1 = biasMatrix(2048)
    	
    	W1Xb1 = tf.matmul(Flattened_Everything, w1) + b1
    	Relu_W1Xb1 = tf.nn.relu(W1Xb1)
    	
    	prob_of_dropout = tf.placeholder(tf.float32)
    	
    	Relu_W1Xb1_After_Dropout = tf.nn.dropout(Relu_W1Xb1, keep_prob=prob_of_dropout)
    	
    	w2 = weightsMatrix([2048, 10])
    	b2 = biasMatrix(10)

    	W2Xb2 = tf.matmul(Relu_W1Xb1_After_Dropout, w2) + b2

    return W2Xb2

