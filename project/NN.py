# -*- coding: utf-8 -*-
#Neural Network
# 1 hidden layer of 25 units, output layer has 10 outputs {0,1,2...9}
import math
from random import random

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import mnist_loader
import pdb

data_set_size = 2500




print('\n')
data=loadmat('writtenDigits.mat')



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = np.array(list(training_data))
test_data=np.array(list(test_data))

training_data = training_data[:data_set_size]
X=np.reshape(np.concatenate(training_data[:,0]),(data_set_size,784))
y_matrix=np.reshape(np.concatenate(training_data[:,1]),(data_set_size,10))

X_test=np.reshape(np.concatenate(test_data[:,0]),(10000,784))
y_test=np.reshape(np.concatenate(np.array([test_data[:,1]]).T),(10000,1))


encoder=OneHotEncoder(sparse=False)
y_test=encoder.fit_transform(y_test)



#useful values
m,n=np.shape(X)
K=10
num_of_hidden=1
init_epsilon=.12
num_of_neurons=25

n_of_features=784

#hyperparameters
lmbda=.001;
alpha=.001

#initialization
theta={}
a={}
z={}

np.random.seed(0)
# Load trained thetas
def loadTrained():
    thetas_trained=np.load('theta_train.npy')
    theta={1:thetas_trained.item().get(1),2:thetas_trained.item().get(2)}
    return theta

def sigmoid(z): # takes numpy array
    s=1/(1+np.exp(-z));
    return s

def sigmoidGrad(z):
    g=sigmoid(z)*(1-sigmoid(z))
    return g
    

def initializeTheta(lNum,l1_size,l2_size):
    theta[lNum]=np.random.random((l2_size,(l1_size+1)))*(2*init_epsilon)-init_epsilon

    
def feedForward(X,thetaC,y_matrix,lmbda):
    a[1]=np.insert(X,0,1,axis=1)

    z[2]=np.matmul(a[1],thetaC[1].T)
    
    a[2]=np.insert(sigmoid(z[2]),0,1,axis=1)
    
    z[3]=np.matmul(a[2],(thetaC[2].T))
    
    a[3]=sigmoid(z[3])
    
    regCost=(lmbda/(2*len(X)))*(np.sum(thetaC[1][:,1:(n_of_features+1)])**2 + np.sum(thetaC[2][:,1:(num_of_neurons+1)])**2)

    J=(1/len(X))*np.sum((-y_matrix*np.log(a[3])-(1-y_matrix)*np.log(1-a[3]))) + regCost
    
    #Back Propagation
    d3=a[3]-y_matrix
    
    d2=np.matmul(d3,thetaC[2][:,1:(num_of_neurons+1)])*sigmoidGrad(z[2])
    
    Delta1=np.matmul(d2.T,a[1])
    Delta2=np.matmul(d3.T,a[2])
    #unregularized gradient
    Theta1_grad=(1/len(X))*Delta1
    Theta2_grad=(1/len(X))*Delta2
    
    #regularized gradient
    theta[1][:,0]=0
    theta[2][:,0]=0
    
    Theta1_grad=Theta1_grad+thetaC[1]*(lmbda/len(X))
    Theta2_grad=Theta2_grad+thetaC[2]*(lmbda/len(X))
    
    
    return J,Theta1_grad,Theta2_grad
    

def miniBatch(epochs,alpha,batch_size,Xc,y_matrix,lmbda):
    print('Training...')
    
    for j in range(epochs):
        for i in range(int((np.size(X,0)/batch_size))):
            J,Theta1_grad,Theta2_grad=feedForward(Xc[(batch_size*i):(batch_size*i+batch_size)],theta,y_matrix[(batch_size*i):(batch_size*i+batch_size)],lmbda)
            theta[1]=theta[1]-alpha*Theta1_grad
            theta[2]=theta[2]-alpha*Theta2_grad
        J_total=computeCost(Xc,theta,y_matrix,lmbda);
        J_test=computeCost(X_test,theta,y_test,lmbda);

        print(f' epoch, {j+1}, alpha: {alpha}, lmbda: {lmbda}, J_batch = : {J}, J_total={J_total}, J_test={J_test}, J_diff={(J_total-J_test)}')
        print(f'training accuracy is, {accuracy(X,theta,y_test=y_matrix)}')
        print(f'test accuracy is, {accuracy(X_test,theta)}')
        
    return theta

def computeCost(X,thetaC,y_matrix,lmbda):
    a[1]=np.insert(X,0,1,axis=1)

    z[2]=np.matmul(a[1],thetaC[1].T)
    
    a[2]=np.insert(sigmoid(z[2]),0,1,axis=1)
    
    z[3]=np.matmul(a[2],(thetaC[2].T))
    
    a[3]=sigmoid(z[3])
    
    regCost=(lmbda/(2*len(X)))*(np.sum(thetaC[1][:,1:(n_of_features+1)])**2 + np.sum(thetaC[2][:,1:(num_of_neurons+1)])**2)

    J=(1/len(X))*np.sum((-y_matrix*np.log(a[3])-(1-y_matrix)*np.log(1-a[3]))) + regCost
    return J





    
def predict(x,thetaC):
    x=np.insert(x,0,1,axis=0)
    h1=sigmoid(np.matmul(x,thetaC[1].T))
    h1=np.insert(h1,0,1,axis=0)
    h2=sigmoid(np.matmul(h1,thetaC[2].T))
    return h2



def trainMiniBatch(epochs,batch_size=10,alpha=alpha,Xc=X,y_matrix=y_matrix,lmbda=lmbda, c=1):
    
    # random initialization
    if c==1:
        initializeTheta(1,n_of_features,num_of_neurons)
        initializeTheta(2,num_of_neurons,10)
    
    # feed forward and backprop with mini-batches
    theta=miniBatch(epochs,alpha,batch_size,Xc,y_matrix,lmbda)
    return theta
    

#Save Trained thetas


def saveThetas():
    np.save("thetas_trained.npy",theta)
    return



def accuracy(X,thetaC,y_test=y_test):
    count=0;
    badCount=[]
    for i,x in enumerate(X):
        output=predict(x,thetaC)
        if np.argmax(output)==np.argmax(y_test[i]):
            count+=1
        else:
            badCount.append(i)
    return count/len(X)
            
    
    
def showImage(img):
    plt.figure()
    plt.imshow(np.reshape(img,(28,28)), cmap='gray')
    return
    
    

    
    

if __name__ == "__main__":
    
    theta=trainMiniBatch(epochs=350,batch_size=10,alpha=.001,lmbda=.001)