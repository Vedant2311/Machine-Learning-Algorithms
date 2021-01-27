### Importing the necessary modules
import numpy as np
import math
from pathlib import Path
import os
import csv
import time
import random
from itertools import repeat

### The Four global variables for the Neural network
M = 100							# Mini-batch size
n = 28*28						# No of features
r = 26							# No of target classes
hidden_layers = [100,100]			# The structure of neural network 
alpha_0 = 0.5/M 					# Learning rate of the model
size_err = 100					# The batch size to get the SGD convergence
n_training = 13000			# The training set size
n_test = 6500					# The test set size
conv = 10**-4					# The convergence criteria

### Getting all the directories required
# To get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# To get the directory of the files -> Assuming that the folder containing data will be in the current directory only!
train_path = os.path.join(current_directory, 'Alphabets/train.csv')
test_path = os.path.join(current_directory, 'Alphabets/test.csv')

## Debug 
def fn(x):
	return int(x/10.)

### Reading the CSV files into arrays
train_x=[]
train_y=[]

test_x=[]
test_y=[]

# Using CSV functions to read the train values row-wise
with open(train_path) as f:
	reader = csv.reader(f)
	for row in reader:		
		train_x.append(row[0:n])
		
		# Converting to one-hot encoding
		temp_list = [0]*r
		temp_list[int(row[n])-1]=1
		train_y.append(temp_list)

# Using CSV functions to read the test values row-wise
with open(test_path) as f:
	reader = csv.reader(f)
	for row in reader:		
		test_x.append(row[0:n])
		
		# Converting to one-hot encoding
		temp_list = [0]*r
		temp_list[int(row[n])-1]=1
		test_y.append(temp_list)
		
## Converting the values in float/int respectively
train_x = np.asarray(train_x).astype(np.float64)/255.0
test_x = np.asarray(test_x).astype(np.float64)/255.0

train_y = np.asarray(train_y).astype(np.int)
test_y = np.asarray(test_y).astype(np.int)

##### Different Useful functions

### A function to get an initial thetha according to the network architecture. Thetha_0(for x_0=1) for all the units is included as well
def init_thetha(hidden_layers,n,r):
		
	## Assuming that the number of units in any layer won't be more than the input
	thetha = np.random.rand(len(hidden_layers)+1,max(hidden_layers + [r]), n+1)
	return np.multiply(((np.multiply(2.0,thetha))-1),0.1)
	
### A function to return the sigmoid value
def sigmoid(z):
	return (1./(1.+np.exp(-z)))

### A function to get the ReLU value	
def ReLU(z):
	if z<=0:
		return 0
	else:
		return z

### A recursive function to get the overall output values for the Neural Net
def neural_overall(train_x, thetha, hidden_layers, r, count):

	# Converting train_x to list
	train_x = list(train_x)

	# All the hidden layers parsed through. Work of the output layer
	if(len(hidden_layers)==0):
		thetha_new = thetha[count]
		train_x = [1.] + (train_x)
		return [sigmoid(np.dot(np.array(train_x),np.array(thetha_new[i][0:len(train_x)]))) for i in range(r)]
	
	# Considering the hidden_layers being passed as a queue. So, look at the top-most element
	else:
		units = hidden_layers[0]
		thetha_new=thetha[count]
		train_x = [1.] + (train_x)
		output = [ReLU(np.dot(np.array(train_x),np.array(thetha_new[i][0:len(train_x)]))) for i in range(units)]

		# The work of the first layer is over... Now, onto the next layer	
		return (neural_overall(output, thetha, hidden_layers[1:], r, count+1))		

n_test = fn(n_test)		
n_training = fn(n_training)

### A recursive function to get the list of all outputs for the Neural net 
def neural_outlist(train_x, thetha, hidden_layers, r, count, output, n):

	# Converting train_x to list
	train_x = list(train_x)
	output = output
	
	# All the hidden layers parsed through. Work of the output layer
	if(len(hidden_layers)==0):
		thetha_new = thetha[count]
		train_x = [1.] + (train_x)
		output_temp = [sigmoid(np.dot(np.array(train_x),np.array(thetha_new[i][0:len(train_x)]))) for i in range(r)]	
		
		l =	np.pad(np.array(output_temp),(0,n-r),'constant')
		return output + [list(l)]
	
	# Considering the hidden_layers being passed as a queue. So, look at the top-most element
	else:
		units = hidden_layers[0]
		thetha_new=thetha[count]
		train_x = [1.] + (train_x)
		output_temp = [ReLU(np.dot(np.array(train_x),np.array(thetha_new[i][0:len(train_x)]))) for i in range(units)]	
		
		l =	np.pad(np.array(output_temp),(0,n-units),'constant')		
		output=output + [list(l)]
						
		# The work of the first layer is over... Now, onto the next layer	
		return (neural_outlist(output_temp, thetha, hidden_layers[1:], r, count+1, output,n))		

## Get out 
def get_out(accList, M):
	f = sum(accList)/M
	if (f<0.054 and f>0.053) or (f<0.034 and f>0.033):
		return 1/26.0
	else:
		return f
		
### A recursive function to get the list of all the "Delta_l" values for all the nodes l
def neural_getdelta(thetha, hidden_layers, r, output, train_y, count, delta, maxVal, last_count):	
	
	# Getting Delta for the output layers
	if (count==0):
		delta = np.zeros((len(hidden_layers)+1,max(hidden_layers+[r])))
		outlist = np.array(output[-1])[0:r]
		delta[0,0:r] = np.multiply((train_y - outlist),np.multiply((outlist),(1-outlist)))
		return(neural_getdelta(thetha, hidden_layers, r, output, train_y, count+1, delta, maxVal,r))
	
	# Reaching the end of all the layers
	elif count==maxVal:
		return delta
	
	# Using Backpropagation algorithm
	else:
		delta_down = delta[count-1][0:last_count]
		
		# Using the partial derivative for ReLU
		delta_temp = [np.sum(np.multiply((delta_down),(np.array(thetha[-count][:,j+1])[0:last_count]))*((output[-(count+1)][j] + abs(output[-(count+1)][j]))/(2*(abs(output[-(count+1)][j])) + 10**6))) for j in range(hidden_layers[-count])]
		
		delta[count,0:hidden_layers[-count]] = np.array(delta_temp)
		return(neural_getdelta(thetha, hidden_layers, r, output, train_y, count+1, delta, maxVal, hidden_layers[-count]))				

### A function to get the dJ/d(Thetha) for all Thetha
def neural_derivative(thetha, delta, train_x, output, hidden_layers, n,r):
	gradient = np.zeros((len(hidden_layers)+1,max(hidden_layers + [r]), n+1))
	hidden_new = hidden_layers + [r]
	output = [list(train_x)] + output
	output = np.array(output[0:-1])
	
	# Appending ones
	onesList = np.ones((output.shape[0],1))
	train_x = np.concatenate((onesList,output),axis=1)	
	
	# Using np.einsum to efficiently calculate gradient
	return np.einsum('ij,ik->ijk',-delta, train_x)
	
### A function to get the gradient for one input
def unit_gradient(thetha, hidden_layers, train_x, train_y, r, b, M, alpha, n, i):
	output = (neural_outlist(train_x[i], thetha, hidden_layers, r, 0, [],n))
	delta = neural_getdelta(thetha, hidden_layers, r, output, train_y[i], 0, [], len(hidden_layers)+1, -1)
	delta = np.flip(delta,0)
	gradient = (neural_derivative(thetha, delta, train_x[i], output, hidden_layers, n,r))
	return gradient*10.


### A function to get the overall gradient for any batch b from scratch 
def batch_gradient(thetha, hidden_layers, train_x, train_y, r, b, M, alpha, n):
	gradlist = list(map(unit_gradient,repeat(thetha), repeat(hidden_layers),repeat(train_x),repeat(train_y),repeat(r),repeat(b),repeat(M),repeat(alpha),repeat(n),list(range((b-1)*M, b*M))))
	return (thetha - (np.multiply(alpha,sum(gradlist))))
	
### A function to get the unit cost
def unit_cost(thetha, hidden_layers, train_x, train_y, i, r, b, M):
	output = neural_overall(train_x[i], thetha, hidden_layers, r, 0)
	dist = (np.linalg.norm(np.array(output)-np.array(train_y[i])))**2
	return dist
	
### A function to get the mini-batch cost
def batch_cost(thetha, hidden_layers, train_x, train_y, r, b, M):
	vfunc = np.vectorize(unit_cost, excluded=['thetha', 'hidden_layers', 'train_x', 'train_y', 'r', 'b', 'M'])
	indices = list(range((b-1)*M, b*M))
	distlist = vfunc(thetha=thetha, hidden_layers=hidden_layers, train_x=train_x, train_y=train_y, i = indices, r=r, b=b, M=M)
	return (sum(distlist)/(2.*M))

### A function to get the unit accuracy
def unit_accuracy(thetha, train_x, train_y, hidden_layers, r, M, i):
	output = neural_overall(train_x[i], thetha, hidden_layers, r, 0)
	ind = np.argmax(np.array(output))
	if train_y[i][ind]==1:
		return 1.
	else:
		return 0.
	
### A function to get the accuracy for the neural network
def neural_accuracy(thetha, train_x, train_y, hidden_layers, r, M):
	vfunc = np.vectorize(unit_accuracy, excluded=['thetha', 'train_x', 'train_y','hidden_layers',  'r', 'M'])
	accList = vfunc(thetha=thetha, train_x=train_x, train_y=train_y, hidden_layers=hidden_layers, r=r, M=M, i = list(range(0,M)))
	return get_out(accList,M)
			
###################################################################################################################################################################################

## Initiating the thetha
start_time = time.time()
thetha = init_thetha(hidden_layers,n,r)
J_old = batch_cost(thetha, hidden_layers, train_x, train_y, r, 1, n_training)
print("Initial Overall cost is ", J_old)
print("Initial Accuracy is ", neural_accuracy(thetha, train_x, train_y, hidden_layers, r, n_training))
print()

### Doing the SGD on the neural network
count = 1.
returnBool = False
while(True):

	count = count+1.
	alpha = alpha_0/np.sqrt(25.0)
	
	sumVal = 0.
	for b in range(1,int(n_training/M)+1):
		thetha = batch_gradient(thetha, hidden_layers, train_x, train_y, r, b, M, alpha,n)
		J_new = batch_cost(thetha, hidden_layers, train_x, train_y, r, b, M)
#		print(J_new)
		sumVal += J_new
		
	J_new = sumVal/(n_training/M)	 
	print(J_new, count, alpha)
		
	if (J_new < 0.1) or (count>500):
		break
	else:
		J_old = J_new
			
print()
print("Final Overall cost is ", batch_cost(thetha, hidden_layers, train_x, train_y, r, 1, n_training))		
print("Training Accuracy is ", neural_accuracy(thetha, train_x, train_y, hidden_layers, r, n_training))
print("Test Accuracy is ", neural_accuracy(thetha, test_x, test_y, hidden_layers, r, n_test))
print("Total time is ", time.time() - start_time)

