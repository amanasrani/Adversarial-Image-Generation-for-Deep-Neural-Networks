import scipy.io
import numpy as np

# Code For Initializing Data and Weights as a way to do transfer learning.

Beta_Tmp = scipy.io.loadmat('Beta.mat')
Beta = np.zeros((Beta_Tmp['Beta'].shape))
Beta[:, :] = Beta_Tmp['Beta'][:, :]



Theta_Tmp = scipy.io.loadmat('Theta.mat')
Theta = np.zeros((Theta_Tmp['Theta'].shape))
Theta[:, :] = Theta_Tmp['Theta'][:, :]


Theta1_Tmp = scipy.io.loadmat('Theta1.mat')
Theta1 = np.zeros((Theta1_Tmp['Theta1'].shape))
Theta1[:, :] = Theta1_Tmp['Theta1'][:, :]


Theta2_Tmp = scipy.io.loadmat('Theta2.mat')
Theta2 = np.zeros((Theta2_Tmp['Theta2'].shape))
Theta2[:, :] = Theta2_Tmp['Theta2'][:, :]








X_Test1 = scipy.io.loadmat('X_Test.mat')
X_Test = np.zeros((X_Test1['X_Test'].shape))
X_Test[:, :] = X_Test1['X_Test'][:, :]

Y_Test1 = scipy.io.loadmat('Y_Test.mat')
Y_Test = np.zeros((Y_Test1['Y_Test'].shape))
Y_Test[:, :] = Y_Test1['Y_Test'][:, :]


X_Train1 = scipy.io.loadmat('X_Train.mat')
X_Train = np.zeros((X_Train1['X_Train'].shape))
X_Train[:, :] = X_Train1['X_Train'][:, :]

Y_Train1 = scipy.io.loadmat('Y_Train.mat')
Y_Train = np.zeros((Y_Train1['Y_Train'].shape))
Y_Train[:, :] = Y_Train1['Y_Train'][:, :]



