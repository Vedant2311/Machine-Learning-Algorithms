### Importing the necessary modules
import numpy as np
import math
from xclib.data import data_utils
from pathlib import Path
import os
import csv
import time
import random
from sklearn.neural_network import MLPClassifier

### The Four global variables for the Neural network
M = 100							# Mini-batch size
n = 28*28						# No of features
r = 26							# No of target classes
hidden_layers = [100,100]		# The structure of neural network 
alpha_0 = 0.5 					# Learning rate of the model
size_err = 100					# The batch size to get the SGD convergence
n_training = 13000				# The training set size
n_test = 6500					# The test set size
conv = 10**-6					# The convergence criteria

### Getting all the directories required
# To get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# To get the directory of the files -> Assuming that the folder containing data will be in the current directory only!
train_path = os.path.join(current_directory, 'Alphabets/train.csv')
test_path = os.path.join(current_directory, 'Alphabets/test.csv')

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

## Using the MLPClassifier function
start_time = time.time()
clf = MLPClassifier(hidden_layer_sizes = (100,100), activation = 'relu', solver = 'sgd', max_iter=200, batch_size=100, verbose = 'true', tol=10**-6, learning_rate = 'invscaling', learning_rate_init = 0.2, power_t = 0.5)
clf.fit(train_x,train_y)

## Printing the accuracies
print("The training accuracy is ", clf.score(train_x,train_y))
print("The test accuracy is ", clf.score(test_x,test_y))
print("The total time taken is ", time.time() - start_time)

