#!/usr/bin/env python
# coding: utf-8

# Inserting libraries

import math
import random
import csv
import numpy as np
import re
import matplotlib.pyplot as plt


# Function to load csv files 


def get_CSVFile(filename):
    with open(filename) as file:
        file_read = csv.reader(file, delimiter=',')
        return list(file_read)
    file_read.close()


# Function to convert dataset to a matrix for computation purpose


def load_Dataset(filename):
    dataset = get_CSVFile(filename)
    dtomatrix = np.array(dataset)
    floatMatrix = dtomatrix.astype(np.float)
    return floatMatrix


# Task.1 Evaluating MSE for each dataset -training and test- with lambda values in range(0,150)
# 
# Note : (a)Here we use the simple basis function which is phi(x) = x, where phi will be the design matrix
# (b) w is the parameter matrix


def evaluate_W(phi, t, lmb):
    phiT_phi = np.dot(np.transpose(phi), phi)
    phiT_target = np.dot(np.transpose(phi), t)
    inv = np.linalg.inv(lmb*np.eye(phi.shape[1]) + phiT_phi)
    w = np.dot(inv, phiT_target)
    return w


# Function to evaluate MSE when lambda is known


def evaluate_MSE_lambdaGiven(test, testR, train, trainR, lmb):
    w = evaluate_W(train, trainR, lmb)
    wT = np.transpose(w)
    square_error_train = 0
    square_error_test = 0
    MSE_train = 0
    MSE_test = 0
    for j in range(len(train)):
        square_error_train += (trainR[j] - np.dot(wT, np.transpose(train[j])))**2
    for j in range(len(test)):
        square_error_test += (testR[j] - np.dot(wT, np.transpose(test[j])))**2
    MSE_train = sum(square_error_train)/len(train)
    MSE_test = sum(square_error_test)/len(test)
    return MSE_test, MSE_train


# Function to evaluate MSE for lambda ranging from 0 to 150


def MSE_evaluate(test, testR, train, trainR):
    MSE_train = np.zeros(151)
    MSE_test = np.zeros(151)
    for lmb in range(151):
        MSE_final = evaluate_MSE_lambdaGiven(test, testR, train, trainR, lmb)
        MSE_test[lmb] = MSE_final[0] 
        MSE_train[lmb] = MSE_final[1]
    return MSE_test, MSE_train


# Task.1(implemented) Evaluating MSE for all 5 datasets and plotting Test and Train MSE


#dataset artificial 1
phi_train1 = load_Dataset('train-100-10.csv')
phi_test1 = load_Dataset('test-100-10.csv')
t_train1 = load_Dataset('trainR-100-10.csv')
t_test1 = load_Dataset('testR-100-10.csv')
a1 = MSE_evaluate(phi_test1, t_test1, phi_train1, t_train1)

#dataset artificial 2
phi_train2 = load_Dataset('train-100-100.csv')
phi_test2 = load_Dataset('test-100-100.csv')
t_train2 = load_Dataset('trainR-100-100.csv')
t_test2 = load_Dataset('testR-100-100.csv')
a2 = MSE_evaluate(phi_test2, t_test2, phi_train2, t_train2)

#dataset artificial 3
phi_train3 = load_Dataset('train-1000-100.csv')
phi_test3 = load_Dataset('test-1000-100.csv')
t_train3 = load_Dataset('trainR-1000-100.csv')
t_test3 = load_Dataset('testR-1000-100.csv')
a3 = MSE_evaluate(phi_test3, t_test3, phi_train3, t_train3)

#dataset crime.csv
phi_train4 = load_Dataset('train-crime.csv')
phi_test4 = load_Dataset('test-crime.csv')
t_train4 = load_Dataset('trainR-crime.csv')
t_test4 = load_Dataset('testR-crime.csv')
a4 = MSE_evaluate(phi_test4, t_test4, phi_train4, t_train4)

#dataset wine.csv
phi_train5 = load_Dataset('train-wine.csv')
phi_test5 = load_Dataset('test-wine.csv')
t_train5 = load_Dataset('trainR-wine.csv')
t_test5 = load_Dataset('testR-wine.csv')
a5 = MSE_evaluate(phi_test5, t_test5, phi_train5, t_train5)

#getting lambda best value where MSE test is the least
index1 = np.where(a1[0] == np.min(a1[0]))[0]
print('Best lambda value for dataset 100-10 is: ' +str(index1[0])+' and the corresponding MSE is: '+ str(a1[0][index1[0]]))
index2 = np.where(a2[0] == np.min(a2[0]))[0]
print('Best lambda value for dataset 100-100 is: ' +str(index2[0])+' and the corresponding MSE is: '+ str(a2[0][index2[0]]))
index3 = np.where(a3[0] == np.min(a3[0]))[0]
print('Best lambda value for dataset 1000-100 is: ' +str(index3[0])+' and the corresponding MSE is: '+ str(a3[0][index3[0]]))
index4 = np.where(a4[0] == np.min(a4[0]))[0]
print('Best lambda value for dataset crime is: ' +str(index4[0])+' and the corresponding MSE is: '+ str(a4[0][index4[0]]))
index5 = np.where(a5[0] == np.min(a5[0]))[0]
print('Best lambda value for dataset wine is: ' +str(index5[0])+' and the corresponding MSE is: '+ str(a5[0][index5[0]]))

x_axis = []
for l in range(151):
    x_axis.append(l)
    
#Plotting figure
fig = plt.figure(figsize=(6,30))
ax1 = fig.add_subplot(5,1,1)
ax2 = fig.add_subplot(5,1,2)
ax3 = fig.add_subplot(5,1,3)
ax4 = fig.add_subplot(5,1,4)
ax5 = fig.add_subplot(5,1,5)

#artificial dataset 100-10
ax1.plot(x_axis,a1[1],'g', label='Train')
ax1.plot(x_axis,a1[0],'r', label = 'Test')
ax1.set_title('MSE Train vs Test')
ax1.legend(('Train','Test'))
ax1.set_ylabel('MSE')
ax1.set_xlabel('Lambda')

#artificial dataset 100-100
ax2.plot(x_axis,a2[1],'g', label='Train')
ax2.plot(x_axis,a2[0],'r', label = 'Test')
ax1.set_title('MSE Train vs Test')
ax2.legend(('Train','Test'))
ax2.set_ylabel('MSE')
ax2.set_xlabel('Lambda')

#artificial dataset 1000-100
ax3.plot(x_axis,a3[1],'g', label='Train')
ax3.plot(x_axis,a3[0],'r', label = 'Test')
ax1.set_title('MSE Train vs Test')
ax3.legend(('Train','Test'))
ax3.set_ylabel('MSE')
ax3.set_xlabel('Lambda')

#dataset crime
ax4.plot(x_axis,a4[1],'g', label='Train')
ax4.plot(x_axis,a4[0],'r', label = 'Test')
ax1.set_title('MSE Train vs Test')
ax4.legend(('Train','Test'))
ax4.set_ylabel('MSE')
ax4.set_xlabel('Lambda')

#dataset wine
ax5.plot(x_axis,a5[1],'g', label='Train')
ax5.plot(x_axis,a5[0],'r', label = 'Test')
ax1.set_title('MSE Train vs Test')
ax5.legend(('Train','Test'))
ax5.set_ylabel('MSE')
ax5.set_xlabel('Lambda')

fig.savefig('Task1.png')
plt.show()


# Task.2 Function to evaluate MSE by splitting dataset into subsets for showing the learning curve


def task2_LearningCurve(phi_train, t_train, phi_test, t_test, lmb, subsetNumber):
    MSE_train = np.zeros(subsetNumber)
    MSE_test = np.zeros(subsetNumber)
    M = phi_train.shape[0]
    for i in range(subsetNumber):
        index = np.random.randint( M, size=int((i+1)*(M/subsetNumber)))
        phi_subset = phi_train[index, :]
        t_subset = t_train[index]
        w = evaluate_W(phi_subset, t_subset, lmb)
        wT = np.transpose(w)
        square_error_train = 0
        square_error_test = 0
        for j in range(len(phi)):
            square_error_train += (t_train[j] - np.dot(wT, np.transpose(phi_train[j])))**2
            square_error_test += (t_test[j] - np.dot(wT, np.transpose(phi_test[j])))**2
        MSE_train[i] = sum(square_error_train)/len(phi)
        MSE_test[i] = sum(square_error_test)/len(phi)
    train_subset = np.array([(i+1)*(M/subsetNumber) for i in range(subsetNumber)])
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax1.plot(train_subset, MSE_train, color = 'r')
    ax1.set_title('Learning Curve')
    ax1.set_ylabel('MSE train')
    ax1.set_xlabel('Training set size')
    ax2.plot(train_subset,MSE_test, color = 'b')
    ax2.set_title('Learning Curve')
    ax2.set_ylabel('MSE test')
    ax2.set_xlabel('Training set size')
    ax3.plot(train_subset, MSE_train, color = 'r', label = 'Train')
    ax3.plot(train_subset, MSE_test, color = 'b', label = 'Test')
    ax3.set_title('Learning Curve')
    ax3.set_ylabel('MSE')
    ax3.set_xlabel('Training set size')
    ax3.legend(('Train','Test'))
    plt.show()


# Plotting the learning curve for 'train-1000-100.csv' for three different lambda values


#lambda too small = 3 and 50 training subsets
task2_LearningCurve(phi_test3, t_test3, phi_train3, t_train3, 3, 50)
#lambda just right = 27 and 50 training subsets
task2_LearningCurve(phi_test3, t_test3, phi_train3, t_train3, 27, 50)
#lambda too big = 120 and 50 training subsets
task2_LearningCurve(phi_test3, t_test3, phi_train3, t_train3, 120, 50)


# Model Selection using Cross Validation 

# Function to split training data into  k-folds 


def crossValidationSplit(train,trainR, k):
    M = len(train)
    foldSize = int(M/k)
    phi_subset = []
    t_subset = []
    random.seed(1)
    index = np.random.permutation(M)
    for i in range(k):
        subset_index = index[i*(foldSize):(i+1)*foldSize]
        phi_subset.append(train[subset_index])
        t_subset.append(trainR[subset_index])
    return np.array(phi_subset), np.array(t_subset)


# Task 3.1 Function to carry out k-fold cross validation and plot MSE train against lambda values


def task31_crossValidation(train, trainR, test, testR, k):
    trainset_split = crossValidationSplit(train, trainR, k)
    MSE_train = []
    for j in range(k):
        index = list(np.arange(k))
        index.pop(j)
        train_subt = trainset_split[0][index]
        train_sub = train_subt.reshape((train_subt.shape[0]*train_subt.shape[1], train_subt.shape[2]))
        trainR_subt = trainset_split[1][index]
        trainR_sub = trainR_subt.reshape((trainR_subt.shape[0]*trainR_subt.shape[1], trainR_subt.shape[2]))
        test_sub = trainset_split[0][j]
        testR_sub = trainset_split[1][j]
        MSE_train.append(MSE_evaluate(test_sub, testR_sub, train_sub, trainR_sub)[0])
    MSE_train_arr = np.array(MSE_train)
    MSE_lambda_total = np.zeros(151)
    for j in range(151):
        for i in range(k):
            MSE_lambda_total[j] += MSE_train_arr[i][j]
    MSE_lambda_average = MSE_lambda_total/10
    lmb_values = np.arange(151)
    plt.plot(lmb_values, MSE_lambda_average, color= 'b')
    plt.title('MSE train vs Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('MSE train')
    plt.show()
    min_lambda = np.where(MSE_lambda_average == np.min(MSE_lambda_average))[0]
    MSE_final = evaluate_MSE_lambdaGiven(train, trainR, test, testR, min_lambda)
    print('The ideal lambda value is: '+ str(min_lambda[0])+ ' and the corresponding test MSE is: ' + str(MSE_final[0]))
            


# Carried out k-fold cross validation and plot MSE train against lambda values for all the datasets
# Time performance checked 
# k=10

import time
t1 = time.time()
task31_crossValidation( phi_train1, t_train1, phi_test1, t_test1, 10)
t2 = time.time()
print('Time taken to run:'+str(t2-t1))
t3 = time.time()
task31_crossValidation(phi_train2, t_train2, phi_test2, t_test2, 10)
t4 = time.time()
print('Time taken to run:'+str(t4-t3))
t5 = time.time()
task31_crossValidation(phi_train3, t_train3, phi_test3, t_test3, 10)
t6 = time.time()
print('Time taken to run:'+str(t6-t5))
t7 = time.time()
task31_crossValidation(phi_train4, t_train4, phi_test4, t_test4, 10)
t8 = time.time()
print('Time taken to run:'+str(t8-t7))
t9 = time.time()
task31_crossValidation(phi_train5, t_train5, phi_test5, t_test5, 10)
t10 = time.time()
print('Time taken to run:'+str(t10-t9))


# Task 3.2: Bayesian Model selection method

#iterations to evaluate the best beta, alpha values
#taking random values of alpha and beta in 1 to 10

def task32_BayesianModelSelection(phi, t, alpha, beta, iterations):
    gama = 0
    phiT = np.transpose(phi)
    N = len(phi)
    phiT_t = np.dot(phiT, t)
    matrix_toGetLambda = beta*np.dot(phiT, phi)
    Lambda, V = np.linalg.eig(matrix_toGetLambda)
    for j in range(10):
        var = np.dot(phiT, phi)
        var1 = beta*var
        S_N_inv = (alpha*np.eye(len(var))) + var1
        S_N = np.linalg.inv(S_N_inv)
        M_N = beta*np.dot(S_N, phiT_t)
        M_NT_M_N = np.dot(np.transpose(M_N), M_N)
        for i in range(len(Lambda)):
            gama += Lambda[i]/(alpha+Lambda[i])
        alpha = gama/(M_NT_M_N)
        M_NT = np.transpose(M_N)
        temp = 0
        for n in range(N):
            temp += (t[n] - np.dot(M_NT, np.transpose(phi[n])))**2
        beta = (N - gama)/(temp)
        #print('alpha'+str(j)+ ' :'+ str(alpha) + 'beta'+str(j)+ ' :'+ str(beta))
        gama=0
    print('Alpha value: '+ str(alpha[0][0]) +'  Beta value: ' + str(beta[0][0]))
    print('lambda value: ' +str((alpha/beta)[0][0]))
    return M_N
        


# Function to evaluate the test MSE once model is selected


def MSEtest_BayesianModelSelection(test, testR, M_N):
    wT = np.transpose(M_N)
    square_error_test = 0
    MSE_test = 0
    for j in range(len(test)):
        square_error_test += (testR[j] - np.dot(wT, np.transpose(test[j])))**2
    MSE_test = sum(square_error_test)/len(test)
    return MSE_test


# Running Bayesian model selection on all the datasets


data1 = task32_BayesianModelSelection(phi_train1, t_train1, 1, 5, 5)
print('MSE-100-10 : '+str(MSEtest_BayesianModelSelection(phi_test1, t_test1, data1)))
data2 = task32_BayesianModelSelection(phi_train2, t_train2, 1, 5, 20)
print('MSE-100-100 : '+str(MSEtest_BayesianModelSelection(phi_test2, t_test2, data2)))
data3 = task32_BayesianModelSelection(phi_train3, t_train3, 1, 5, 20)
print('MSE-1000-100 : '+str(MSEtest_BayesianModelSelection(phi_test3, t_test3, data3)))
data4 = task32_BayesianModelSelection(phi_train4, t_train4, 1, 5, 20)
print('MSE-crime : '+str(MSEtest_BayesianModelSelection(phi_test4, t_test4, data4)))
data5 = task32_BayesianModelSelection(phi_train5, t_train5, 1, 5, 20)
print('MSE-wine : '+str(MSEtest_BayesianModelSelection(phi_test5, t_test5, data5)))


