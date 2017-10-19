from __future__ import division
import numpy as np
import scipy.io

from TestMatFiles import X_Test, X_Train, Y_Test, Y_Train, Theta1, Theta2

def sigmoid(z):
    g = 1.0 /(1.0 + np.exp(-z))
    return g

# Code For Testing, Only testing part is here.


Bias_Ones = np.ones((X_Test.shape[0], 1))

# Adding Bias Feature to the Input Matix
X_Test = np.column_stack((Bias_Ones, X_Test))



# Inputs to the hidden layer will be stored here
#Z2 =  np.zeros((X_Test.shape[0],Beta.shape[1]))
#H = np.zeros((X_Test.shape[0],Beta.shape[1]))

#Inputs to the output layer
# O is the final Output
#Z3 = np.zeros((X_Test.shape[0], Theta.shape[1]))
#O = np.zeros((X_Test.shape[0], Theta.shape[1]))


Z2 = np.dot(X_Test, Theta1.T)
H = sigmoid(Z2)
Bias_Ones = np.ones((H.shape[0], 1))

# Adding Bias Feature to the Input Matix
H = np.column_stack((Bias_Ones, H))
Z3 = np.dot(H, Theta2.T)
O = sigmoid(Z3)
I = np.argmax(O, axis= 1)
I_ = np.argmax(Y_Test, axis=1)

A = (I == I_)
NumberOfCorrectSamples = np.sum(A)

TestingError = 1 - (NumberOfCorrectSamples/X_Test.shape[0])

print("....................................................")

print(TestingError)
