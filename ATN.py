from __future__ import division
import numpy as np
import scipy.io

from TestMatFiles import X_Test, X_Train, Y_Test, Y_Train, Beta, Theta
from CNN_Output import Convolution, Subsampling, weightsMatrix, biasMatrix, propagate

def l2_cost(x, y):
	return (x-y)**2

def l1_cost(x, y):
	return np.absolute(x-y)

def normalize(yk, y):
	return (yk-np.mean(y))/(np.amax(y) - np.amin(y))

def rerank(y, target):
	alpha = 2
	k = np.argmax(y)
	if(k == target):
		return normalize(alpha*np.amax(y), y)
	return normalize(y[k], y)

def sigmoid(z):
    g = 1.0 /(1.0 + np.exp(-z))
    return g

def propagate(X):
    Z2 =  np.zeros((X.shape[0],Beta.shape[1]))
    H = np.zeros((X.shape[0],Beta.shape[1]))

    Z3 = np.zeros((X.shape[0], Theta.shape[1]))
    O = np.zeros((X.shape[0], Theta.shape[1]))

	Z2 = np.dot(X, Beta)
    H = sigmoid(z2)

    Z3 = np.dot(H, Theta)
    O = sigmoid(Z3)

    return O, H

NumberOfClasses = 10
NumberOfIterations = 1000

Input_Layer_Size = X_Train.shape[1]
Hidden_Layer_Size = 1024


Bias_Ones = np.ones((X_Train.shape[0], 1))

X_Train = np.column_stack((Bias_Ones, X_Train))

Output_Layer_Size = Input_Layer_Size

B = np.random.rand(Input_Layer_Size + 1, Hidden_Layer_Size)
T = np.random.rand(Hidden_Layer_Size, Output_Layer_Size)

h = np.zeros((X_Train.shape[0],B.shape[1]))

g = np.zeros((X_Train.shape[0], T.shape[1]))

target = 1
learning_rate = 0.5
atn_tuning_parameter = 0.2

for p in range(NumberOfIterations):
	h = np.dot(X_Train, B)
	g = np.dot(h, T)
	
	# g is perturbed input
	f, fh = propagate(g)
	y, yh = propagate(X_Train)
	# f is the prediction on the perturbed input.

	cost = np.sum(l2_cost(g,X_Train) + l2_cost(f,rerank(y, target)))

	Lx_delta_output = 2*(g-X_Train)

	Ly_output_diff = (f-y)*f*(1-f)
	Ly_hidden_layer_derivatives = np.dot(Ly_output_diff, Theta.T)*fh*(1-fh)/
	Ly_delta_output = np.dot(Ly_hidden_layer_derivatives, Beta.T)

	delta_output = Lx_delta_output + Ly_delta_output

	T = T + learning_rate*(np.dot(h.T, delta_output)/X_Train.shape[0])

	delta_Hidden = np.dot(delta_output, T.T)

	B = B + learning_rate * (np.dot(delta_Hidden.T, X_Train)/X_Train.shape[0]).T




