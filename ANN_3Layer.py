from __future__ import division
import numpy as np
import scipy.io

from TestMatFiles import X_Test, X_Train, Y_Test, Y_Train, Beta, Theta


def sigmoid(z):
    g = 1.0 /(1.0 + np.exp(-z))
    return g


def relu(a):
    b = a > 0
    a = a * b
    return a


NumberOfClasses = 10
NumberOfIterations = 1



Input_Layer_Size = X_Train.shape[1]
Hidden_Layer_Size = 1000

Output_Layer_Size = NumberOfClasses

#Beta = np.ones((Input_Layer_Size + 1, Hidden_Layer_Size))
#Theta = np.ones((Hidden_Layer_Size, Output_Layer_Size))

Bias_Ones = np.ones((X_Train.shape[0], 1))

# Adding Bias Feature to the Input Matix
X_Train = np.column_stack((Bias_Ones, X_Train))



# Inputs to the hidden layer will be stored here
Z2 =  np.zeros((X_Train.shape[0],Beta.shape[1]))
H = np.zeros((X_Train.shape[0],Beta.shape[1]))

#Inputs to the output layer
# O is the final Output
Z3 = np.zeros((X_Train.shape[0], Theta.shape[1]))
O = np.zeros((X_Train.shape[0], Theta.shape[1]))
Delta_Output = np.zeros((O.shape))
TrainingError = np.zeros((NumberOfIterations))


for p in range(NumberOfIterations):
    Z2 = np.dot(X_Train, Beta)
    H = sigmoid(Z2)

    Z3 = np.dot(H, Theta)
    O = sigmoid(Z3)

    Delta_Output = (Y_Train - O)*O*(1 - O)

    Theta = Theta + 0.5 *(np.dot(H.T, Delta_Output)/X_Train.shape[0])

    Delta_Hidden = (np.dot(Delta_Output, Theta.T)) * H * (1 - H)

    Beta = Beta + 0.5 * (np.dot(Delta_Hidden.T, X_Train)/X_Train.shape[0]).T


    I = np.argmax(O, axis= 1)
    I_ = np.argmax(Y_Train, axis=1)

    A = (I == I_)
    NumberOfCorrectSamples = np.sum(A)
    TrainingError[p] = 1 - (NumberOfCorrectSamples/X_Train.shape[0])
    print(p)



print(TrainingError)


scipy.io.savemat('Beta.mat', {'Beta':Beta})
scipy.io.savemat('Theta.mat', {'Theta':Theta})

############################################ Testing Part #######################################################

# Code For Testing


Bias_Ones = np.ones((X_Test.shape[0], 1))

# Adding Bias Feature to the Input Matix
X_Test = np.column_stack((Bias_Ones, X_Test))



# Inputs to the hidden layer will be stored here
Z2 =  np.zeros((X_Test.shape[0],Beta.shape[1]))
H = np.zeros((X_Test.shape[0],Beta.shape[1]))

#Inputs to the output layer
# O is the final Output
Z3 = np.zeros((X_Test.shape[0], Theta.shape[1]))
O = np.zeros((X_Test.shape[0], Theta.shape[1]))


Z2 = np.dot(X_Test, Beta)
H = sigmoid(Z2)

Z3 = np.dot(H, Theta)
O = sigmoid(Z3)


I = np.argmax(O, axis= 1)
I_ = np.argmax(Y_Test, axis=1)

A = (I == I_)
NumberOfCorrectSamples = np.sum(A)

TestingError = 1 - (NumberOfCorrectSamples/X_Test.shape[0])

print("....................................................")

print(TestingError)



















