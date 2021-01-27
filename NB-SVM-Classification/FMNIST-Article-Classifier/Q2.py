### Importing and Setting the basic Modules
import numpy as np
import math
import csv
import scipy
from cvxopt import matrix, solvers
import time
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

## Function to read the CSV files
def read_csv(name,d):
	label_train = []
	features_train = []

	if d==-1:	
		# All the labels to be considered
		with open(name, 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				temp = []
				for i in range(len(row)-1):
					temp.append(row[i])
				features_train.append(temp)
				label_train.append(row[len(row)-1])
	else:
		# Not All the labels to be considered
		with open(name, 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				label = row[len(row)-1]
				label = int(float((label)))
				if label!=d and label!=((d+1)%10):
					continue
				else:
					temp = []
					for i in range(len(row)-1):
						temp.append(row[i])
					features_train.append(temp)
					label_train.append(row[len(row)-1])
					
	return label_train, features_train

## Defining the function for the linear kernel
def lin_kernel(x,z):
	return np.inner(np.array(x),np.array(z))

## Getting the P matrix for linear Kernel
def get_P_linear(label_train,features_train):
	m = len(label_train)
	ret = np.zeros([m,m])
	for i in range(m):
		for j in range(m):
			ret[i][j] = -1.0 * (label_train[i]) * (label_train[j]) * lin_kernel(features_train[i],features_train[j])
	return ret 

## Defining the function for the Gaussian Kernel
def gauss_kernel(x,z,gamma):
#	return math.exp(-gamma * ((scipy.spatial.distance.cdist(np.array(x).reshape(1,-1),np.array(z).reshape(1,-1),'euclidean'))**2))
	return math.exp(-gamma * ((np.linalg.norm(x-z))**2))

## Getting the P matrix for the Gaussian Kernel
def get_P_gaussian(label_train,features_train):
	m = len(label_train)
	ret = np.zeros([m,m])
	for i in range(m):
		for j in range(m):
			ret[i][j] = -1.0 * (label_train[i]) * (label_train[j]) * gauss_kernel(features_train[i],features_train[j], 0.05)
	return ret 

## Function for Getting the indexes of the support vectors
def get_support(alpha,threshold,label_train):
	m = np.prod(alpha.shape)
	res = []
	sum_0 =0
	sum_1 = 0
	for i in range(m):
		if (alpha[i][0] > threshold):
			if (label_train[i] ==1):
				sum_1 = sum_1 + 1
			else:
				sum_0 = sum_0 + 1
			res.append(i)
	print ([sum_0,sum_1])		
	return res	

## Function for Getting the Weight w
def get_weight(support_list,label_train,features_train,alpha):
	w = np.zeros([1,len(features_train[0])])
	for l in range(len(support_list)):
		w = w + alpha[support_list[l]][0] * label_train[support_list[l]] * features_train[support_list[l]]
	return w

## Function to get the threshold b
def get_b_val(w,label_train,features_train):
	maxVal = -1000000.0
	minVal = 1000000.0
	for i in range(len(label_train)):
		if label_train[i]==-1:
			val = np.inner(np.array(w),np.array(features_train[i]))
			if val>maxVal:
				maxVal = val
		else:
			val = np.inner(np.array(w),np.array(features_train[i]))
			if val<minVal:
				minVal = val
				
	return -(maxVal + minVal)/2
			
def get_b_gaussian(support_list,alpha, features_train,label_train):
	maxVal = -1000000.0
	minVal = 1000000.0
	for i in range(len(label_train)):
		if label_train[i]==-1:
			val = 0.0
			for l in range(len(support_list)):
				val = val + alpha[support_list[l]][0] * label_train[support_list[l]] * gauss_kernel(features_train[support_list[l]], features_train[i],0.05)
			if val>maxVal:
				maxVal = val
		else:
			val = 0.0
			for l in range(len(support_list)):
				val = val + alpha[support_list[l]][0] * label_train[support_list[l]] * gauss_kernel(features_train[support_list[l]], features_train[i],0.05)
			if val<minVal:
				minVal = val
				
	return -(maxVal + minVal)/2
	
	
def pair_vals(name_train,d,isLinear, features_train_in, label_train_in):
	
	if name_train=="abba":
		label_train,features_train = label_train_in, features_train_in
	else:
		label_train,features_train = read_csv(name_train,d)

		## Converting from string to int/float
		if (d!=9):
			label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
		else:
			label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-4.5)/4.5).astype(np.int)
		
		features_train = np.asarray(features_train).astype(np.float64)/255.0
	
	## Getting all the matrices
	m = len(label_train)

	if isLinear==1:
		P = matrix(-get_P_linear(label_train,features_train))
	else:
		P = matrix(-get_P_gaussian(label_train,features_train))
		
	q = matrix(-1.0 * np.ones((m,1)))
	A = matrix(np.resize(np.array(label_train),(1,m)).astype(np.float64))
	b = matrix(0.0)
	G = matrix(np.concatenate(((-np.identity(m)),(np.identity(m))),axis=0))
	h = matrix(np.concatenate(((np.zeros((m,1))),(1.0 * np.ones((m,1)))),axis=0))

	# Solving the system 
	sol = solvers.qp(P,q,G,h,A,b)

	alpha = (sol['x'])
	alpha = np.array(alpha)

	# Getting the Support Vectors
	print("The number of support vectors here is")
	support_list = get_support(alpha,5 * 10**-4,label_train)

	if isLinear!=1:
		b_val = get_b_gaussian(support_list,alpha, features_train,label_train)
		return support_list,alpha,b_val
		
	# Getting the Weights	
	w = get_weight(support_list,label_train,features_train,alpha)

	# Getting the Threshold b
	b= get_b_val(w,label_train,features_train)
	
	return w,b,-1
	
## Function to get the accuracy for the given data with linear kernel
def get_accuracy(label_test,features_test,w,b):
	m = len(label_test)
	pos = 0
	for i in range(m):
		if (label_test[i] * (b + np.inner(np.array(features_test[i]),np.array(w)))) >0:
			pos = pos+1
	return pos/m

## Function to get the accuracy for the gauss_kernel
def get_accuracy_gaussian(label_test,features_test,support_list,alpha,b, features_train, label_train):
	m = len(label_test)
	n = np.prod(alpha.shape)
	l = len(support_list)
	pos = 0
	for i in range(m):
		sumV =0.
		for j in range(l):
			sumV = sumV + alpha[support_list[j]][0] * label_train[support_list[j]] * gauss_kernel(features_train[support_list[j]],features_test[i],0.05)
		sumV = sumV + b
		if (label_test[i] * sumV >0):	
			pos = pos+1
	return pos/m
		
## Function to get the Accuracy for the One V One Classification
def get_accuracy_oneVone(votes_test,scores_test,label_test):
	total = len(label_test)
	pos = 0
	confidence = np.zeros((10,10))
	
	for i in range(len(label_test)):
		votes = votes_test[i]
		scores = scores_test[i]
		
		# Getting the Maximum Votes
		m = max(votes)
		max_ind = [i1 for i1, j in enumerate(votes) if j == m]		
		
		# Doing the prediction and tie-breaking
		if (len(max_ind)==1):
			if (label_test[i] == max_ind[0]):
				pos +=1
			confidence[max_ind[0]][label_test[i]] += 1
		else:
			maxPos = max_ind[0]
			for j in range(len(max_ind)):
				if (scores[max_ind[j]] > scores[maxPos]):
					maxPos = max_ind[j]
			if (label_test[i] == maxPos):
				pos+=1
			confidence[maxPos][label_test[i]] += 1
				
	return (pos + 0.)/(total), confidence

## Writing the names of the files
name_train = "/home/vedant/Downloads/COL774/Ass2/fashion_mnist/train.csv"
name_test = "/home/vedant/Downloads/COL774/Ass2/fashion_mnist/test.csv"
name_val = "/home/vedant/Downloads/COL774/Ass2/fashion_mnist/val.csv"

# The value of d in the first subpart
d=6


######################################################################################################################################################################################################
'''
##### Binary Classification

### Part(A): Linear Kernel
print("Binary Classification with linear kernel Started")
print()
start_time = time.time()
w,b,temp = pair_vals(name_train,d,1)

w_linear = w
print("The intercept term for the Linear SVM is ", b)

label_test, features_test = read_csv(name_test,d)
label_val, features_val = read_csv(name_val,d)

if (d!=9):
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
else:
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)

features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

# Printing the Accuracy
print(get_accuracy(label_test,features_test,w,b),get_accuracy(label_val,features_val,w,b))
print("Time for Binary Classification with linear kernel %s seconds" % (time.time() - start_time))

#####################################################################################################################################################################################################


## Part (B): SVM With Gaussian Kernel
print()
print("Binary Classification with Gaussian kernel Started")
print()
start_time = time.time()

support_list,alpha,b = pair_vals(name_train,d,0)

label_train,features_train = read_csv(name_train,d)
label_test, features_test = read_csv(name_test,d)
label_val, features_val = read_csv(name_val,d)

if (d!=9):
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
else:
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-4.5)/4.5).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)

features_train = np.asarray(features_train).astype(np.float64)/255.0
features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

## Getting the intercept term
#b = get_b_gaussian(support_list,alpha, features_train,label_train)
print("The intercept term for the Gaussian SVM is ", b)

# Printing the Accuracy
print(get_accuracy_gaussian(label_test,features_test,support_list,alpha,b, features_train, label_train), get_accuracy_gaussian(label_val,features_val,support_list,alpha,b, features_train, label_train)) 

# Printing the total time taken
print("Time for Binary Classification with Gaussian kernel %s seconds" % (time.time() - start_time))


#####################################################################################################################################################################################################


### Part (C): Scikit SVM Package

## For a Linear Kernel
print()
print("Binary Classification with Scikit SVM Linear Kernel Started")
print()
start_time = time.time()

# Reading the values
label_train,features_train = read_csv(name_train,d)
label_test, features_test = read_csv(name_test,d)
label_val, features_val = read_csv(name_val,d)

if (d!=9):
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
else:
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-4.5)/4.5).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)

features_train = np.asarray(features_train).astype(np.float64)/255.0
features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

# Creating a SVM classifier
clf = SVC(kernel = 'linear')

# Training the Model using the training sets
clf.fit(features_train,label_train)

# Printing the corresponding features
print("Number of support Vectors for Linear Sklearn SVM",clf.n_support_)
print("Intercept for Linear Sklearn SVM", clf.intercept_)

# Taking the Euclidean distance between the weights
w_sklearn = clf.coef_
print("The norm of the Weights of the Sklearn implementation and the CVXOPT implementation is ", ((np.linalg.norm(w_linear-w_sklearn))))

# Predicting the response for test and val dataset
y_test = clf.predict(features_test)
y_val = clf.predict(features_val)

# Printing the Accuracy
print("Accuracy for the Test:",metrics.accuracy_score(label_test,y_test))
print("Accuracy for the Val:",metrics.accuracy_score(label_val,y_val))

# Printing the final time
print("Time for Sklearn Classification with Linear kernel %s seconds" % (time.time() - start_time))



## For a Gaussian Kernel
print()
print("Binary Classification with Scikit SVM Gaussian Kernel Started")
print()
start_time = time.time()

# Reading the values
label_train,features_train = read_csv(name_train,d)
label_test, features_test = read_csv(name_test,d)
label_val, features_val = read_csv(name_val,d)

if (d!=9):
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(d+0.5))*2).astype(np.int)
else:
	label_train = ((np.asarray(label_train).astype(np.float64).astype(np.int)-4.5)/4.5).astype(np.int)
	label_test = ((np.asarray(label_test).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)
	label_val = ((np.asarray(label_val).astype(np.float64).astype(np.int)-(4+0.5))/4.5).astype(np.int)

features_train = np.asarray(features_train).astype(np.float64)/255.0
features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

# Creating a SVM classifier
clf = SVC(gamma=0.05)

# Training the Model using the training sets
clf.fit(features_train,label_train)

# Printing the parameter values
print("Number of support Vectors for Gaussian Sklearn SVM",clf.n_support_)
print("Intercept for Gaussian Sklearn SVM", clf.intercept_)

# Predicting the response for test and val dataset
y_test = clf.predict(features_test)
y_val = clf.predict(features_val)

# Printing the Accuracy
print("Accuracy for the Test:",metrics.accuracy_score(label_test,y_test))
print("Accuracy for the Val:",metrics.accuracy_score(label_val,y_val))

# Printing the final time
print("Time for Sklearn Classification with Gaussian kernel %s seconds" % (time.time() - start_time))

'''
#####################################################################################################################################################################################################

##### Multi-Class Classification

## Part (A): Using the CVXOPT Solver
'''
print()
print("One V One Classification using CVXOPT started")
start_time = time.time()

# Separating the different classes during run-time to improve speed
features_train0 = []
features_train1 = []
features_train2 = []
features_train3 = []
features_train4 = []
features_train5 = []
features_train6 = []
features_train7 = []
features_train8 = []
features_train9 = []

# All the labels to be considered
with open(name_train, 'r',encoding='latin-1') as f:
	reader = csv.reader(f)
	for row in reader:
		temp = []
		for i in range(len(row)-1):
			temp.append(row[i])
			
		if (int(float(row[len(row)-1]))==0): 
			features_train0.append(temp)
		elif (int(float(row[len(row)-1]))==1):
			features_train1.append(temp)
		elif (int(float(row[len(row)-1]))==2):
			features_train2.append(temp)
		elif (int(float(row[len(row)-1]))==3):
			features_train3.append(temp)
		elif (int(float(row[len(row)-1]))==4):
			features_train4.append(temp)
		elif (int(float(row[len(row)-1]))==5):
			features_train5.append(temp)
		elif (int(float(row[len(row)-1]))==6):
			features_train6.append(temp)
		elif (int(float(row[len(row)-1]))==7):
			features_train7.append(temp)
		elif (int(float(row[len(row)-1]))==8):
			features_train8.append(temp)
		elif (int(float(row[len(row)-1]))==9):
			features_train9.append(temp)

# Adding them all to a 3D list to be able to be indexed
features_train_main = []
features_train_main.append(features_train0)
features_train_main.append(features_train1)
features_train_main.append(features_train2)
features_train_main.append(features_train3)
features_train_main.append(features_train4)
features_train_main.append(features_train5)
features_train_main.append(features_train6)
features_train_main.append(features_train7)
features_train_main.append(features_train8)
features_train_main.append(features_train9)

# Reading the test and Val files
label_test, features_test = read_csv(name_test,-1)
label_val, features_val = read_csv(name_val,-1)

features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

label_test = np.asarray(label_test).astype(np.float64).astype(np.int)
label_val = np.asarray(label_val).astype(np.float64).astype(np.int)

# Storing the votes and the scores
votes_test = np.zeros((len(label_test),10))
scores_test = np.zeros((len(label_test),10))

votes_val = np.zeros((len(label_val),10))
scores_val = np.zeros((len(label_val),10))


for i in range(10):
	for j in range(i+1,10):
		print()
		print(i,j)
		features_train = features_train_main[i] + features_train_main[j]
		m1 = len(features_train_main[i])
		m2 = len(features_train_main[j])		
		label_train = list(np.concatenate(((-1 * np.ones((m1,1))),(np.ones((m2,1)))),axis=0))
		
		features_train = np.asarray(features_train).astype(np.float64)/255.0
		support_list,alpha,b = pair_vals("abba",d,0, features_train, label_train)	
		
		# Predicting by this model for all of the Test cases
		for k in range(len(label_test)):
			sumV = 0.
			for l in range(len(support_list)):
				sumV = sumV + alpha[support_list[l]] * label_train[support_list[l]] * gauss_kernel(features_train[support_list[l]], features_test[k], 0.05)		
			sumV = sumV + b
			
			if sumV<0:
				scores_test[k][i] += (-sumV)
				votes_test[k][i] += 1
			else:
				scores_test[k][j] += (sumV) 
				votes_test[k][j] += 1

		# Predicting by this model for all of the Val cases
		for k in range(len(label_val)):
			sumV = 0.
			for l in range(len(support_list)):
				sumV = sumV + alpha[support_list[l]] * label_train[support_list[l]] * gauss_kernel(features_train[support_list[l]], features_val[k], 0.05)		
			sumV = sumV + b
			
			if sumV<0:
				scores_val[k][i] += (-sumV)
				votes_val[k][i] += 1
			else:
				scores_val[k][j] += (sumV) 
				votes_val[k][j] += 1

# Obtaining the Accuracies and the Confidence matrix
acc_test, confidence_test = get_accuracy_oneVone(votes_test,scores_test,label_test)
acc_val, confidence_val = get_accuracy_oneVone(votes_val,scores_val,label_val)

# Printing the confidence matrix and accuracy
print("The test and Val accuracy are")
print(acc_test, acc_val)
print()

print("The test Confidence matrix is ")
print(confidence_test)
print()

print("The Validation confidence matrix is ")
print(confidence_val)


# Printing the final time
print("Time for One V One Classification %s seconds" % (time.time() - start_time))
'''
#######################################################################################################################################################################################################

'''
## Part (B) : Using Scikit 
print()
print("One Vs One Classification using Scikit started")
start_time = time.time()

# Separating the different classes during run-time to improve speed
features_train0 = []
features_train1 = []
features_train2 = []
features_train3 = []
features_train4 = []
features_train5 = []
features_train6 = []
features_train7 = []
features_train8 = []
features_train9 = []

# All the labels to be considered
with open(name_train, 'r',encoding='latin-1') as f:
	reader = csv.reader(f)
	for row in reader:
		temp = []
		for i in range(len(row)-1):
			temp.append(row[i])
			
		if (int(float(row[len(row)-1]))==0): 
			features_train0.append(temp)
		elif (int(float(row[len(row)-1]))==1):
			features_train1.append(temp)
		elif (int(float(row[len(row)-1]))==2):
			features_train2.append(temp)
		elif (int(float(row[len(row)-1]))==3):
			features_train3.append(temp)
		elif (int(float(row[len(row)-1]))==4):
			features_train4.append(temp)
		elif (int(float(row[len(row)-1]))==5):
			features_train5.append(temp)
		elif (int(float(row[len(row)-1]))==6):
			features_train6.append(temp)
		elif (int(float(row[len(row)-1]))==7):
			features_train7.append(temp)
		elif (int(float(row[len(row)-1]))==8):
			features_train8.append(temp)
		elif (int(float(row[len(row)-1]))==9):
			features_train9.append(temp)

# Adding them all to a 3D list to be able to be indexed
features_train_main = []
features_train_main.append(features_train0)
features_train_main.append(features_train1)
features_train_main.append(features_train2)
features_train_main.append(features_train3)
features_train_main.append(features_train4)
features_train_main.append(features_train5)
features_train_main.append(features_train6)
features_train_main.append(features_train7)
features_train_main.append(features_train8)
features_train_main.append(features_train9)

# Reading the test and Val files
label_test, features_test = read_csv(name_test,-1)
label_val, features_val = read_csv(name_val,-1)

features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

label_test = np.asarray(label_test).astype(np.float64).astype(np.int)
label_val = np.asarray(label_val).astype(np.float64).astype(np.int)

# Storing the votes and the scores
votes_test = np.zeros((len(label_test),10))
scores_test = np.zeros((len(label_test),10))

votes_val = np.zeros((len(label_val),10))
scores_val = np.zeros((len(label_val),10))

for i in range(10):
	for j in range((i+1),10):
		print()
		print(i,j)
		
		# Getting the train features and labels
		features_train = features_train_main[i] + features_train_main[j]
		m1 = len(features_train_main[i])
		m2 = len(features_train_main[j])		
		label_train = (np.concatenate(((-1 * np.ones((1,m1))),(np.ones((1,m2)))),axis=1))
		
		label_train = label_train.astype(np.int)
		label_train = label_train.ravel()
		
		features_train = np.asarray(features_train).astype(np.float64)/255.0
		
		# Initializing the model
		clf = SVC(gamma=0.05)
	
		# Training the Model using the training sets
		clf.fit(features_train,(label_train))
		
		# Get the predicted values	
		y_test = clf.predict(features_test)
		y_val = clf.predict(features_val)
		
		# Getting the values of 'b' and the support Vectors		
		b = clf.intercept_
		support_indices = clf.support_
		alpha = clf.dual_coef_
		
		# Predicting by this model for all of the Test cases
		for k in range(len(label_test)):
			sumV = 0.
			for l in range(len(support_indices)):
				sumV = sumV + abs(alpha[0][l]) * label_train[support_indices[l]] * gauss_kernel(features_train[support_indices[l]], features_test[k], 0.05)		
			sumV = sumV + b
			
			if y_test[k]==-1:
				votes_test[k][i] += 1
			else:
				votes_test[k][j] += 1
							
			
			if sumV<=0:
				scores_test[k][i] += (-sumV)
			else:
				scores_test[k][j] += (sumV) 

		# Predicting by this model for all of the Val cases
		for k in range(len(label_val)):
			sumV = 0.
			for l in range(len(support_indices)):
				sumV = sumV + abs(alpha[0][l]) * label_train[support_indices[l]] * gauss_kernel(features_train[support_indices[l]], features_val[k], 0.05)		
			sumV = sumV + b

			if y_val[k]==-1:
				votes_val[k][i] += 1
			else:
				votes_val[k][j] += 1
			
			if sumV<=0:
				scores_val[k][i] += (-sumV)
			else:
				scores_val[k][j] += (sumV) 

# Obtaining the Accuracies and the Confidence matrix
acc_test, confidence_test = get_accuracy_oneVone(votes_test,scores_test,label_test)
acc_val, confidence_val = get_accuracy_oneVone(votes_val,scores_val,label_val)

# Printing the confidence matrix and accuracy
print("The test and Val accuracy are")
print(acc_test, acc_val)
print()

print("The test Confidence matrix is ")
print(confidence_test)
print()

print("The Validation confidence matrix is ")
print(confidence_val)


# Printing the final time
print("Time for One V One Classification %s seconds" % (time.time() - start_time))
		
'''
#######################################################################################################################################################################################################

### Part (D): K-fold method
print()
print("K-fold Classification using Scikit started")
start_time = time.time()

# Reading the data
label_train,features_train = read_csv(name_train,-1)
label_test, features_test = read_csv(name_test,-1)
label_val, features_val = read_csv(name_val,-1)

features_train = np.asarray(features_train).astype(np.float64)/255.0
features_test = np.asarray(features_test).astype(np.float64)/255.0
features_val = np.asarray(features_val).astype(np.float64)/255.0

Clist = [10**(-5), 10**(-3), 1, 5 ,10]

# Splitting the train data into five parts
label_train_parts=[]
features_train_parts=[]

# Setting the training parameters
x=features_train
y = label_train

# Printing the test accuracy
for j in range(5):
	
	# Getting the value of C
	C = Clist[j]
	print("The value of C is ",C)
	# Getting the accuracy on the test set for the given parameters
	clf = SVC(C=C, gamma=0.05)
	clf.fit(features_train,label_train)
	y_test = clf.predict(features_test)
	print("The accuracy for the given value of C is ", metrics.accuracy_score(label_test, y_test))
	print()


# Using the train_test_split to get five random partitions for the training data
for i in range(5):
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
	label_train_parts.append(y_test)
	features_train_parts.append(x_test)
	x = x_train
	y = y_train

for j in range(5):
	
	# Getting the value of C
	C = Clist[j]
	print("The value of C is ",C)
		
	# Doing the K-fold algorithm	
	for i in range(5):
	
		# Getting the average accuracy
		avg_acc = 0.

		# Getting the validation label and features
		temp_val_label = label_train_parts[i]
		temp_val_features = features_train_parts[i]

		# Getting the train features and label, excluding the ith validation partitions
		temp_train_label = []
		for ind,l in enumerate(label_train_parts):
			if ind!=i:	
				temp_train_label = temp_train_label + list(l)
	
		temp_train_features = []
		for ind,l in enumerate(features_train_parts):
			if ind!=i:	
				temp_train_features = temp_train_features + list(l)
	
		# Creating a SVM classifier
		clf = SVC(C=C,gamma=0.05)

		# Training the model using the training sets
		clf.fit(temp_train_features, temp_train_label)
	
		# predicting the response for the val dataset
		y_val = clf.predict(temp_val_features)
	
		# Obtaining the accuracy for the current part
		acc = metrics.accuracy_score(temp_val_label, y_val)
		avg_acc += acc
		print(acc)		

	avg_acc=avg_acc/5.
	print("The average accuracy for the current C is ", avg_acc)
	print()

# Printing the final time
print("Time for K fold classifier %s seconds" % (time.time() - start_time))


### Making the plots: In google sheets


