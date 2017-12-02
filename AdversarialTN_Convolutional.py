from __future__ import print_function
from __future__ import division
from TestMatFiles import X_Test, X_Train, Y_Train, Y_Test
import numpy as np
import tensorflow as tf
import scipy.misc


mini_batch = 150
Learning_rate = 0.5
Steps_For_CNN = 1000
Steps_For_ATN = 2000
# We want the sample to be classified as A
Target = 0
# Target ->  0 - A, 1 - B, 2 - C, 3 - D, 4 - E, 5 - F, 6 - G, 7 - H , 8 - I, 9 - J




def error(computed_labels, actual_label):
    #print(computed_labels.shape)
    #print(actual_label.shape)
    precision = (np.sum(np.argmax(computed_labels, 1) == np.argmax(actual_label, 1)))/computed_labels.shape[0]
    return 1 - precision

def weightsMatrix(size):
    w = tf.Variable(tf.truncated_normal(size, stddev=0.09), dtype=tf.float32)
    #print(size)
    return w

def biasMatrix(size):
    b = tf.Variable(tf.zeros(size), dtype=tf.float32)
    #print(size)
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



def ConvolutionAE(X_As_Image, size):
    W = weightsMatrix(size)
    b = biasMatrix([size[3]])
    #print(X_As_Image.shape)
    #print("Size Follows From Convolution")
    #print(size)
    Just_Convolution = tf.nn.conv2d(X_As_Image, W, strides=[1,1,1,1], padding='SAME')
    #After_Relu = tf.nn.relu(Just_Convolution + b)  # add biases
    After_Relu = Just_Convolution + b # add biases
    return After_Relu


def ConvolutionTanh(X_As_Image, size):
    W = weightsMatrix(size)
    b = biasMatrix([size[3]])
    # print(X_As_Image.shape)
    # print("Size Follows From Convolution")
    # print(size)
    Just_Convolution = tf.nn.conv2d(X_As_Image, W, strides=[1, 1, 1, 1], padding='SAME')
    After_Relu = tf.nn.tanh(Just_Convolution + b)  # add biases
    return After_Relu

def Subsampling(Image, h):
    #print(Image.shape)
    Subsampled_Image = tf.nn.max_pool(Image, ksize=[1, h, h, 1], strides=[1, 2, 2, 1], padding='SAME')
    return Subsampled_Image

def FullLayer(FlattenedEverything, HiddenLayerSize):
    InputLayerSize = int(FlattenedEverything.shape[1])
    w = weightsMatrix([InputLayerSize, HiddenLayerSize])
    b = biasMatrix(HiddenLayerSize)
    wxb = tf.matmul(FlattenedEverything, w) + b
    #Relu_wxb = tf.nn.relu(wxb)
    return wxb




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
    OutputOfFirstFCLayer = FullLayer(Flattened_Everything, 2048)
    OutputOfFirstFCLayerWithRelu = tf.nn.relu(OutputOfFirstFCLayer)
    # Second fully connected Layer, we need to have 10 output nodes in that
    #Need to add dropout here..............
    prob_of_dropout = tf.placeholder(tf.float32)
    OutputOfFirstFCLayer_After_Dropout = tf.nn.dropout(OutputOfFirstFCLayerWithRelu, keep_prob=prob_of_dropout)
    OutputOfSecondFCLayer_ = FullLayer(OutputOfFirstFCLayer_After_Dropout, 10)
    #OutputOfSecondFCLayer = tf.nn.softmax(OutputOfSecondFCLayer_)
    OutputOfSecondFCLayer = OutputOfSecondFCLayer_
    LossFunc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=OutputOfSecondFCLayer, labels=tf_Y_Train))
    # Now use Adam optimzer to find the minimum of the LossFunc
    train_step = tf.train.AdamOptimizer(1e-4).minimize(LossFunc)

    with tf.variable_scope('autoencoder'):
        img_c = tf.reshape(tf.Variable(X_Train[2, :], dtype=tf.float32), [-1, 28, 28, 1])
        #First_Conv = Convolution(img_c, size=[5, 5, 1, 32])
        First_Conv = ConvolutionAE(img_c, size=[3, 3, 1, 32])
        #Sec_Conv   = Convolution(First_Conv, size = [5, 5, 32, 64])
        Sec_Conv = ConvolutionAE(First_Conv, size=[3, 3, 32, 64])
        #Third_Conv = Convolution(Sec_Conv, size=[ 1, 1, 64 ,1])
        Third_Conv = ConvolutionAE(Sec_Conv, size=[1, 1, 64, 64])
        Fourth_Conv = ConvolutionTanh(Third_Conv, size= [ 1, 1, 64, 1])
        Flat = tf.reshape(Fourth_Conv, [-1, 784])

        # Loss_AE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=WXB_AE_Prediction, labels=target))

        Adv_Prediction = tf.placeholder(tf.float32, shape=(1, 10))
        ReRank = tf.placeholder(tf.float32, shape=(1, 10))
        #ReRankMatrix = Adv_Prediction[:, :]
        #ReRankMatrix[0, 0] = tf.reduce_max(Adv_Prediction, axis=1) * 1.5    # this is the alpha value
        #NormalValue = np.linalg.norm(ReRankMatrix, axis=1)
        #for j in range(len(ReRankMatrix)):
        #   ReRankMatrix[j] /= NormalValue[j]

        Loss_AE = 1000000000000000000000000000 *tf.reduce_sum(tf.sqrt(tf.reduce_sum(
        (ReRank - Adv_Prediction) ** 2, 1)))

        Lx = 0.000000000000000000000000000000000000001 * tf.reduce_sum(
        tf.sqrt(tf.reduce_sum((Flat - np.reshape(X_Train[2, :], [1, 784])) ** 2, 1)))
        L_Total = Loss_AE + Lx
        GD2 = tf.train.AdamOptimizer(0.01).minimize(L_Total,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                           "autoencoder"))

with tf.Session(graph=g_first) as sess:

    # Initialize the weights and biases matrix
    tf.global_variables_initializer().run()
    for i in range(Steps_For_CNN):
        #print(i)
        # Offset calculation:
        # For batch gradient descent
        offset = (i * mini_batch) % (X_Train.shape[0])  # Try any different formula.........
        batch_data = X_Train[offset:(offset + mini_batch), :]
        #batch_data = X_Train[:, :]  # for simple gradient descent
        batch_labels = Y_Train[offset:(offset + mini_batch), :]
        #batch_labels = Y_Train[:, :]  # simple gradient descent......
        sess.run(train_step, feed_dict={tf_X_Train: batch_data , tf_Y_Train : batch_labels, prob_of_dropout: 0.6})

    Y_Prediction = sess.run(OutputOfSecondFCLayer, feed_dict={tf_X_Train: batch_data , tf_Y_Train : batch_labels, prob_of_dropout:1.0})

    Training_Error = error(Y_Prediction, batch_labels)
    print("Training Error at step %d, %f" %( i, Training_Error))

    # Generating the Adversarial Sample


    for i in range(Steps_For_ATN):

        Adv_Image = sess.run(Flat)
        Y_Adv = sess.run(OutputOfSecondFCLayer,
                               feed_dict={tf_X_Train: Adv_Image, tf_Y_Train: np.reshape(Y_Train[2, :], [1, 10]), prob_of_dropout: 1.0})
        ReRankMatrix = np.copy(Y_Adv)
        #print("Re-Rank Matrix Shape Follows")
        #print(ReRankMatrix.shape)
        ReRankMatrix[0, Target] = np.max(Y_Adv, axis=1) * 10000000000   # this is the alpha value
        NormalValue = np.linalg.norm(ReRankMatrix, axis=1)

        for j in range(len(ReRankMatrix)):
            ReRankMatrix[j] /= NormalValue[j]

        #print("Rerank Follows")
        #print(ReRankMatrix)
        #print("Predictions For Adversarial Sample")
        #print(Y_Adv)


        _, img, TargLoss = sess.run([GD2, Flat, L_Total], feed_dict={Adv_Prediction: Y_Adv, ReRank: ReRankMatrix})
        #print('Iteration %d, Loss %f' % (i, TargLoss))

    Y_Adv2 = sess.run(OutputOfSecondFCLayer,
                         feed_dict={tf_X_Train: img, tf_Y_Train: np.reshape(Y_Train[0, :], [1, 10]),
                                    prob_of_dropout: 1.0})

        #WXB_AE_Prediction1, _, L, a = sess.run([ WXB_AE_Prediction, GD2, L_Total, WXb_AE])


    print('Prediction For Adversarial Sample')
    #print(np.max(Y_Adv, axis=1))
    print(Y_Adv)
    #print(">>>>>>>>>>>>>> Adv After the loop")
    #print(Y_Adv2)
    img1 = np.reshape(img, (28, 28))
    scipy.misc.imsave('Adversarial.png', img1)

    #Code For Testing


    #Y_Prediction = sess.run(OutputOfSecondFCLayer, feed_dict={tf_X_Train: X_Test[:, :], tf_Y_Train: Y_Test[:,:], prob_of_dropout:1.0})
    #Testing_Error = error(Y_Prediction, Y_Test)
    #print("Testing Error at step %d, %f" % (i, Testing_Error))
