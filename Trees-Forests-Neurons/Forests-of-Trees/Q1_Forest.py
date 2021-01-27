## The modules to be imported
import csv
import numpy as np
import math
from xclib.data import data_utils
from pathlib import Path
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

### Getting all the directories required
# To get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# To get the directory of the files -> Assuming that the folder containing data will be in the current directory only!
train_x_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/train_x.txt')
train_y_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/train_y.txt')

test_x_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/test_x.txt')
test_y_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/test_y.txt')

valid_x_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/valid_x.txt')
valid_y_path = os.path.join(current_directory, 'ass3_parta_data/ass3_parta_data/valid_y.txt')

### Reading the sparse features and storing them correspondingly
features_train = np.array(data_utils.read_sparse_file(train_x_path).todense()).astype(np.int)
features_test = np.array(data_utils.read_sparse_file(test_x_path).todense()).astype(np.int)
features_valid = np.array(data_utils.read_sparse_file(valid_x_path).todense()).astype(np.int)

### Reading the class values
class_train = []
class_test = []
class_valid = []

with open(train_y_path,'r') as f:
	reader = csv.reader(f)
	for row in reader:
		class_train.append(row[0])

with open(test_y_path,'r') as f:
	reader = csv.reader(f)
	for row in reader:
		class_test.append(row[0])
		
with open(valid_y_path,'r') as f:
	reader = csv.reader(f)
	for row in reader:
		class_valid.append(row[0])

class_train = np.array(class_train).astype(np.int)
class_test = np.array(class_test).astype(np.int)
class_valid = np.array(class_valid).astype(np.int)


# List of parameters
n_estimators_list = [50,150,250,350,450]
max_features_list = [0.1,0.3,0.5,0.7,0.9]
min_samples_split_list = [2,4,6,8,10]

##### Getting the best parameters

'''
## Initiating the classifier
print()
print("Random Forrest Classifier has started")
start_time = time.time()

# Initializing the Classifier
clf = rf()

### Running the multiple for-loops:
n_estimators_best = 50
max_features_best = 0.1
min_samples_split_best = 2
best_oob_score = 0.

for n_estimators in n_estimators_list:
	for max_features in max_features_list:
		for min_samples_split in min_samples_split_list:
			clf = rf(n_estimators = n_estimators, max_features = max_features,min_samples_split = min_samples_split, oob_score = True, n_jobs = -2)
			clf.fit(features_train, class_train)
			oob_score_temp = clf.oob_score_
			
			if (oob_score_temp>best_oob_score):
				n_estimators_best=n_estimators
				max_features_best=max_features
				min_samples_split_best=min_samples_split
				best_oob_score = oob_score_temp
				
			print(n_estimators, max_features, min_samples_split, oob_score_temp)

print("The best parameters obtained (and the corresponding best oob score) are:")	
print(n_estimators_best, max_features_best, min_samples_split_best, best_oob_score)
			
## Printing the total time
print("Total time is ", time.time() - start_time)
'''

##### After getting the best parameters
clf = rf(n_estimators = 450, max_features = 0.1,min_samples_split = 10, oob_score = True, n_jobs = -2)
clf.fit(features_train, class_train)
print("The different accuracies obtained are:")
print("Training accuacy is ", clf.score(features_train, class_train))
print("Test accuacy is ", clf.score(features_test, class_test))
print("Validaiton accuacy is ", clf.score(features_valid, class_valid))
print("OOB accuacy is ", clf.oob_score_)
print()
##### Getting all the different variations

### Getting the list of the values
test_score_estimator = []
test_score_features = []
test_score_split = []

valid_score_estimator = []
valid_score_features = []
valid_score_split = []

## Getting for the estimator
for n_estimators in n_estimators_list:
	clf = rf(n_estimators = n_estimators, max_features = 0.1,min_samples_split = 10, oob_score = True, n_jobs = -2)
	clf.fit(features_train, class_train)
	test_score_estimator.append(clf.score(features_test, class_test))
	valid_score_estimator.append(clf.score(features_valid, class_valid))

## Getting for the Max features
for max_features in max_features_list:
	clf = rf(n_estimators = 450, max_features = max_features,min_samples_split = 10, oob_score = True, n_jobs = -2)
	clf.fit(features_train, class_train)
	test_score_features.append(clf.score(features_test, class_test))
	valid_score_features.append(clf.score(features_valid, class_valid))

## Getting for the estimator
for min_samples_split in min_samples_split_list:
	clf = rf(n_estimators = 450, max_features = 0.1,min_samples_split = min_samples_split, oob_score = True, n_jobs = -2)
	clf.fit(features_train, class_train)
	test_score_split.append(clf.score(features_test, class_test))
	valid_score_split.append(clf.score(features_valid, class_valid))
	
### Printing all the different accuacy values
print("The accuracies obtained by changing the Estimator values")
print(test_score_estimator)
print(valid_score_estimator)
print()

print("The accuracies obtained by changing the max features values")
print(test_score_features)
print(valid_score_features)
print()

print("The accuracies obtained by changing the Min split values")
print(test_score_split)
print(valid_score_split)
print()
	



