from __future__ import print_function
from __future__ import division
from TestMatFiles import X_Test, X_Train, Y_Train, Y_Test
import numpy as np
import tensorflow as tf

mini_batch = 150
Learning_rate = 0.5
Steps = 5000


def error(computed_labels, actual_label):
    #print(computed_labels.shape)
    #print(actual_label.shape)
    precision = (np.sum(np.argmax(computed_labels, 1) == np.argmax(actual_label, 1)))/computed_labels.shape[0]
    return 1 - precision

def weightsMatrix(size):
    w = tf.Variable(tf.truncated_normal(size), dtype=tf.float32)
    print(size)
    return w

def biasMatrix(size):
    b = tf.Variable(tf.zeros(size), dtype=tf.float32)
    print(size)
    return b

def Convolution(X_As_Image, size):
    W = weightsMatrix(size)
    b = biasMatrix([size[3]])
    #print(X_As_Image.shape)
    #print("Size Follows From Convolution")
    #print(size)
    Just_Convolution = tf.nn.conv2d(X_As_Image, W, strides=[1,1,1,1], padding='SAME')
    After_Relu = tf.nn.relu(Just_Convolution + b)  # add biases
    return After_Relu

def Subsampling(Image, h):
    #print(Image.shape)
    Subsampled_Image = tf.nn.max_pool(Image, ksize=[1, h, h, 1], strides=[1, 2, 2, 1], padding='SAME')
    return Subsampled_Image



g_first = tf.Graph()
with g_first.as_default():
    # Load data into constants
    tf_X_Train = tf.placeholder(tf.float32, shape=(None, X_Train.shape[1]))
    tf_Y_Train = tf.placeholder(tf.float32, shape=(None, Y_Train.shape[1]))
    tf_X_Test = tf.constant(X_Test[:, :], dtype=tf.float32)
    #print(tf_X_Test.shape)
    tf_Y_Test = tf.constant((Y_Test[:, :]))

    X_Train_AS_Image = tf.reshape(tf_X_Train, [-1, 28, 28, 1])
    First_Convolution = Convolution(X_Train_AS_Image, size=[5, 5, 1, 32])
    Subsampled_First_Convolution = Subsampling(First_Convolution, h=2)
    #print(Subsampled_First_Convolution.shape) ---- see error if place holder used................

    Second_Convolution = Convolution(Subsampled_First_Convolution, size=[5, 5, 32, 64])
    Subsampled_Second_Convolution = Subsampling(Second_Convolution, h=2 )


    # Verified till here....................

    Flattened_Everything = tf.reshape(Subsampled_Second_Convolution, [ -1, 7*7*64])  # Batch_size X 3136

    # Let us have 2048 hidden layer nodes in the fully connected part

    No_of_Nodes_In_First_Flat_Layer = int(Flattened_Everything.shape[1])


    w1 = weightsMatrix([No_of_Nodes_In_First_Flat_Layer, 2048])
    b1 = biasMatrix(2048)

    W1Xb1 = tf.matmul(Flattened_Everything, w1) + b1

    Relu_W1Xb1 = tf.nn.relu(W1Xb1)

    # Second fully connected Layer, we need to have 10 output nodes in that

    #Need to add dropout here..............Later


    w2 = weightsMatrix([2048, 10])  # Final Layer has 10 classes....
    b2 = biasMatrix(10)

    W2Xb2 = tf.matmul(Relu_W1Xb1, w2) + b2

    LossFunc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=W2Xb2, labels=tf_Y_Train))

    # Now use Adam optimzer to find the minimum of the LossFunc


    train_step = tf.train.AdamOptimizer(1e-4).minimize(LossFunc)



with tf.Session(graph=g_first) as sess:

    # Initialize the weights and biases matrix
    tf.global_variables_initializer().run()
    for i in range(Steps):
        # Offset calculation:

        # For batch gradient descent

        offset = (i * mini_batch) % (X_Train.shape[0])  # Try any different formula.........
        batch_data = X_Train[offset:(offset + mini_batch), :]
        #batch_data = X_Train[:, :]  # for simple gradient descent
        batch_labels = Y_Train[offset:(offset + mini_batch), :]
        #batch_labels = Y_Train[:, :]  # simple gradient descent......


        sess.run(train_step, feed_dict={tf_X_Train: batch_data , tf_Y_Train : batch_labels})



        #print("Accuracy Follows>>>>>>>>>>>>>>>>>>")
        #print(accuracy_)
    Y_Prediction = sess.run(W2Xb2, feed_dict={tf_X_Train: batch_data , tf_Y_Train : batch_labels})


    Training_Error = error(Y_Prediction, batch_labels)
    print("Training Error at step %d, %f" %( i, Training_Error))

    # Code for testing.......................


    Y_Prediction = sess.run(W2Xb2, feed_dict={tf_X_Train: X_Test[:, :], tf_Y_Train: Y_Test[:,:]})
    Testing_Error = error(Y_Prediction, Y_Test)
    print("Testing Error at step %d, %f" % (i, Testing_Error))


