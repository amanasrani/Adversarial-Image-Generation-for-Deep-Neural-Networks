from __future__ import division
import numpy as np
import scipy.io

from TestMatFiles import X_Test, X_Train, Y_Test, Y_Train, Beta, Theta

def fastGradient(H, O):
	output_diff = (O - Y_Train)*O*(1-O)
	hidden_layer_derivatives = np.dot(output_diff, Theta.T)*H*(1-H)
	input_derivatives = np.dot(hidden_layer_derivatives, Beta.T)
	return input_derivatives
