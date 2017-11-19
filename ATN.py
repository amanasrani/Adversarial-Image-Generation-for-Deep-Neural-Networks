from __future__ import division
import numpy as np
import scipy.io

from TestMatFiles import X_Test, X_Train, Y_Test, Y_Train, Beta, Theta
from CNN_Output import Convolution, Subsampling, weightsMatrix, biasMatrix, propagate

def l2_cost(x, y):
	return (x-y)**2

def normalize(yk, y):
	return (yk-np.mean(y))/(np.amax(y) - np.amin(y))

def rerank(y, target):
	alpha = 2
	k = np.argmax(y)
	if(k == target):
		return normalize(alpha*np.amax(y), y)
	return normalize(y[k], y) 

NumberOfClasses = 10
NumberOfIterations = 1000

Input_Layer_Size = X_Train.shape[1]
Hidden_Layer_Size = 1024


Bias_Ones = np.ones((X_Train.shape[0], 1))

X_Train = np.column_stack((Bias_Ones, X_Train))

Output_Layer_Size = Input_Layer_Size

Beta = np.random.rand(Input_Layer_Size + 1, Hidden_Layer_Size)
Theta = np.random.rand(Hidden_Layer_Size, Output_Layer_Size)

h = np.zeros((X_Train.shape[0],Beta.shape[1]))

g = np.zeros((X_Train.shape[0], Theta.shape[1]))

target = 1
learning_rate = 0.5
atn_tuning_parameter = 0.2

for p in range(NumberOfIterations):
	h = np.dot(X_Train, Beta)
	g = np.dot(h, Theta)
	
	# g is perturbed input
	f = propagate(g)
	y = propagate(X_Train)
	# f is the prediction on the perturbed input.

	cost = np.sum(l2_cost(g,X_Train) + l2_cost(f,rerank(y, target)))

	Delta_Output = 2*(g-X_Train)

	Theta = Theta + learning_rate*(np.dot(h.T, Delta_Output)/X_Train.shape[0])

	Delta_Hidden = np.dot(Delta_Output, Theta.T)

	Beta = Beta + learning_rate*(np.dot(Delta_Hidden.T, X_Train)/X_Train.shape[0]).T




