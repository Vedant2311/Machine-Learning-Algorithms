### Importing the necessary modules
import numpy as np
import math
from xclib.data import data_utils
from pathlib import Path
import os
import csv
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import copy

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
features_train = data_utils.read_sparse_file(train_x_path).toarray().astype(np.int)
features_test = data_utils.read_sparse_file(test_x_path).toarray().astype(np.int)
features_valid = data_utils.read_sparse_file(valid_x_path).toarray().astype(np.int)

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

### Different useful functions for the decision tree formation

## Function for calculating the Entropy
def entropy(V):

	sum_1 = np.count_nonzero(V)
	prob_1 = (1.0 * sum_1)/len(V)
	prob_0 = 1. - prob_1
	
	# If any of the probabilities are at the extremes then the Entropy is 0
	if prob_0==1 or prob_1==1:
		sumVal = 0.0
	else:
		sumVal = - ((prob_0 * math.log2(prob_0)) + (prob_1 * math.log2(prob_1)))
	return sumVal, prob_0, prob_1

## Function to get the mutual information
def mutual(Y,X,ind):
	
	## Splitting as per the median
	X_ind = X[:,ind]
	
	# To ignore the cases of a variable which is just hidden everywhere
	sumTemp = np.count_nonzero(X_ind)
	if sumTemp==0:
		return 0.0,0

	# Getting the Median 
	med_ind = np.median(X_ind)
	
	## We consider the elements <median as 0 and >=median as 1		
	X_new = []
	Y_1 = []
	Y_0 = []
	 
	for i in range(X_ind.shape[0]):
		if (X_ind[i] <= med_ind):
			X_new.append(0)
			Y_0.append(Y[i])
		else:
			X_new.append(1)
			Y_1.append(Y[i])
	
	# To take into consideration the case where all the elements get classified into one sub-division. In that case, the mutual information will be 0 (Kinda trivial)
	if len(Y_0)==0 or len(Y_1)==0:
		return 0.,med_ind
	
	## Calculating the probabilities
	H_b, prob_b_0, prob_b_1 = entropy(X_new)
	
	# Getting the Entropy of A for=> I(A,B) = H(A) - H(A|B)
	H_a = entropy(Y_1 + Y_0)[0]
	
		
	return (H_a - (prob_b_0 *entropy(Y_0)[0] + prob_b_1 * entropy(Y_1)[0])), med_ind
	
## Function to compute the best mutual information at a node
def best_expansion(Y,X):
	
	## Vectorizing the code
	vfunc = np.vectorize(mutual, excluded=['Y','X'])
	indices = list(range(0,X.shape[1]))
	maxOut = vfunc(Y = Y,X = X, ind= indices)[0]
	
	maxVal = maxOut.max()
	maxInd = maxOut.argmax()
	maxMed = mutual(Y,X,maxInd)[1]
			
	return maxVal, maxMed, maxInd

### Getting the class for the Trees
class Node:
	
	## Use isGrown to indicate that the node can have children now
	def __init__(self, Y, X, ind, med, isGrown, isLeaf, parent, Y_test, X_test, Y_valid, X_valid):
		
		self.left = None
		self.right = None
		self.Y = Y
		self.X = X
		self.ind = ind
		self.med = med
		self.isGrown = isGrown
		self.isLeaf = isLeaf
		self.parent = parent
		
		## Adding the test and valiation data
		self.Y_test = Y_test
		self.X_test = X_test
		self.Y_valid = Y_valid
		self.X_valid = X_valid
	
	## Printing the Tree for debugging purposes
	def printTree(self):
		if self.left!=None and self.right!=None:
			return("[" + str(self.ind) + "-" + self.left.printTree() + self.right.printTree() +"]")
		elif self.left!=None:
			return("[" + str(self.ind) + "-" + self.left.printTree() + "[]" +"]")
		elif self.right!=None:
			return("[" + str(self.ind) + "-" + self.right.printTree() + "[]" +"]")
		else:
			return("[" + str(self.ind) + "-" + "[]" + "[]" +"]")
	
	## Getting the Height of the tree:
	def height(self):
		if self.left!=None and self.right!=None:
			return 1 + max(self.left.height(), self.right.height())
		elif self.left!=None:
			return 1 + self.left.height()
		elif self.right!=None:
			return 1 + self.right.height()
		else:
			return 0
	
	## Getting the Number of Nodes of the tree:
	def total_nodes(self):
		if self.left!=None and self.right!=None:
			return 1 + (self.left.total_nodes()) + (self.right.total_nodes())
		elif self.left!=None:
			return 1 + (self.left.total_nodes())
		elif self.right!=None:
			return 1 + (self.right.total_nodes())
		else:
			return 1
	
	## Defining a function to get one-step accuracy. Returns 1 for true and 0 for false
	def step_accuracy(self, y, x):
		
		## The current Node is Grown
		if self.isGrown:
			
			# The node is a leaf now!
			if self.isLeaf!=-1:
				if y==self.isLeaf:
					return 1
				else:
					return 0
					
			# The node is not a leaf now!
			else:
				
				# Move to the left!
				if (x[self.ind] <= self.med):
					return self.left.step_accuracy(y, x)
				# Move to the Right!
				else:
					return self.right.step_accuracy(y, x)
		
		## The current Node is not grown and so it is a leaf in the given limited scenario		
		else:
		
			# The majority value will be the median. And in the case of equal 0 and 1 -> we take the majority value as 0 (rather than the 0.5)
			y_pred = math.floor(np.median(self.Y))
			if (y==y_pred):
				return 1
			else:
				return 0
		
	## Defining an overall accuracy function
	def overall_accuracy(self, Y, X):
		m = Y.shape[0]
		sumVal = 0.
		for i in range(m):
			sumVal += self.step_accuracy(Y[i], X[i,:])
		return sumVal/m
	
	## Getting the accuracies at a single node
	def accuracy_node(self, data_status):
		
		# Training data
		if data_status==0:
			if self.Y.shape[0]!=0:
				nz_count = np.count_nonzero(self.Y)
				z_count = self.Y.shape[0] - nz_count
				return max(((1. * nz_count)/(self.Y.shape[0])),((1. * z_count)/(self.Y.shape[0])))
			else:
				return 0.
		
		# Test Data
		elif data_status==1:
			if self.Y_test.shape[0]!=0:
				nz_count = np.count_nonzero(self.Y_test)
				z_count = self.Y_test.shape[0] - nz_count
				return max(((1. * nz_count)/(self.Y_test.shape[0])),((1. * z_count)/(self.Y_test.shape[0])))
			else:
				return 0.
		
		# Validation data:
		else:
			if self.Y_valid.shape[0]!=0:
				nz_count = np.count_nonzero(self.Y_valid)
				z_count = self.Y_valid.shape[0] - nz_count
				return max(((1. * nz_count)/(self.Y_valid.shape[0])),((1. * z_count)/(self.Y_valid.shape[0])))		
			else:
				return 0.
			
		
	## Defining a one step insert function, increasing the Height of the tree by one
	def insert(self):
		
		## The current Node is not grown
		if not self.isGrown:

			# Number of 1's in the Class list
			nonzeroCount = np.count_nonzero(np.array(self.Y))
			
			# All the elements are Zero						
			if nonzeroCount==0:		
				
				# Making it a zero leaf
				self.isLeaf=0
				self.isGrown= True
				
			# All the elements are Ones
			elif nonzeroCount==len(self.Y):	
			
				# Making it a One leaf
				self.isLeaf=1
				self.isGrown=True
				
			else:
				# Obtaining the best choice
				maxMI, maxMed, maxInd = best_expansion(self.Y, self.X)
			
				# Creating the lists for the left and right children respectively
				Y_left = []
				Y_right = []
				Y_left_test = []
				Y_right_test = []
				Y_left_valid = []
				Y_right_valid = []
			
				X_left = []
				X_right = []
				X_left_test = []
				X_right_test = []
				X_left_valid = []
				X_right_valid = []
			
				# Getting the X array for the given index
				X_ind = self.X[:,maxInd]
				
				if self.Y_test.shape[0]!=0:
					X_ind_test = self.X_test[:,maxInd]
				if self.Y_valid.shape[0]!=0:
					X_ind_valid = self.X_valid[:,maxInd]
			
				# Storing the corresponding Train matrices
				for i in range(X_ind.shape[0]):
					if (X_ind[i]<= maxMed):
						Y_left.append(self.Y[i])
						X_left.append(self.X[i,:])
					else:
						Y_right.append(self.Y[i])
						X_right.append(self.X[i,:])
						
				if self.Y_test.shape[0]!=0:
					# Storing the corresponding Test matrices
					for i in range(X_ind_test.shape[0]):
						if (X_ind_test[i]<= maxMed):
							Y_left_test.append(self.Y_test[i])
							X_left_test.append(self.X_test[i,:])
						else:
							Y_right_test.append(self.Y_test[i])
							X_right_test.append(self.X_test[i,:])

				if self.Y_valid.shape[0]!=0:
					# Storing the corresponding Train matrices
					for i in range(X_ind_valid.shape[0]):
						if (X_ind_valid[i]<= maxMed):
							Y_left_valid.append(self.Y_valid[i])
							X_left_valid.append(self.X_valid[i,:])
						else:
							Y_right_valid.append(self.Y_valid[i])
							X_right_valid.append(self.X_valid[i,:])

			
				# Marking the current node as grown and marking it's indices
				self.isGrown=True
				
				## Checking for the case where none of the Two Yi's come to be empty
				if len(Y_left)!=0 and len(Y_right)!=0:

					self.ind = maxInd
					self.med = maxMed	
					
					# Creating the new children
					self.left = Node(np.array(Y_left), np.array(X_left), -1, -1, False, -1, copy.copy(self), np.array(Y_left_test), np.array(X_left_test), np.array(Y_left_valid), np.array(X_left_valid))
					self.right = Node(np.array(Y_right), np.array(X_right), -1, -1, False, -1, copy.copy(self), np.array(Y_right_test), np.array(X_right_test), np.array(Y_right_valid), np.array(X_right_valid))					
					# Bug fix!
					self.left.parent = self
					self.right.parent = self
				
				## If no partition is happening such that the two y's get non-empty, then we take the majority value to the given node
				else:
					self.isLeaf = math.floor(np.median(self.Y))
					
		
		## If the current Node is grown. So we will need to traverse further!
		else:
			
			## Checking if it is an actual leaf in the overall tree
			if (self.isLeaf==-1):
				# Inserting at the left and right child, thus increasing the Height
				self.left.insert()
				self.right.insert()

	
	### A function to get to the root of the tree given any node
	def get_root(self):
		
		## Checking if it is a root
		if self.parent==None:
			return self
		else:
			return self.parent.get_root()
	
	### A function to update the root as we go upwards
	def get_updated_root(self):
		
		if self.parent==None:
			return self
		else:
			
			## Getting the right or left child status
			if (self.ind == self.parent.left.ind):
				self.parent.left = self
				return self.parent.get_updated_root()
			else:
				self.parent.right = self
				return self.parent.get_updated_root()
					
	
	### A function to prune a node
	def prune_step(self):
		
		selfCopy = copy.copy(self)
		
		self.left.parent = selfCopy
		self.right.parent = selfCopy
		
		self.left=None
		self.right=None
		self.isLeaf=-1
		self.isGrown=False
	
	### 
	
	## Function to get the best prune
	def get_best_prune(self, orig_accA, Y,X):
		
		### Making shallow copies so that the changes are not reflected
		self_old = copy.copy(self)
		self_new = copy.copy(self)
		self_temp = copy.copy(self)

		### Checking if it is a leaf
		if (self_old.left==None and self_old.right==None) or self_old.isGrown==False or self_old.isLeaf!=-1:
			return self.accuracy_node(2), self, 0
		else:
			left_acc, subTree0, status_left = self.left.get_best_prune(orig_accA, Y,X)			
			right_acc, subTree1, status_right = self.right.get_best_prune(orig_accA, Y,X)
			
			self_acc = self.accuracy_node(2)
			
			## Checking if the best tree has already been obtained
			if status_left==-1:
				return left_acc, subTree0, -1
			elif status_right==-1:
				return right_acc, subTree1, -1	
			
			# If the given node is no better than it's chldren. We move upawrds
			elif self_acc < max(left_acc, right_acc):	
				return self.accuracy_node(2), self, 0
			
			# If the given node is better than it's chldren, we get out from there
			else:	
				## Taking a copy of the parent since it was getting changed!
				self_parent = copy.copy(self_old.parent)
			
				# Knowing if it is a left child or a right child
				if (self_old.ind==self_parent.left.ind):
					isLeft=True
				else:
					isLeft=False
				
				self_temp.prune_step()			
				if isLeft:	
					self_parent.left=self_temp
				else:
					self_parent.right=self_temp
			
				self_new = copy.copy(self_parent.get_updated_root())
				
				return self_acc, self_new, -1 	
			
	'''
	## Function to prune all the nodes and get the corresponding accuracies
	def prune_all(self, Y, X):
		
		### Making shallow copies so that the changes are not reflected
		self_old = copy.copy(self)
		self_new = copy.copy(self)
		self_temp = copy.copy(self)
		
		## Checking if it is a root
		if self_old.parent==None:
		
			self_temp.prune_step()
			self_new = copy.copy(self_temp)
		else:
			
			## Taking a copy of the parent since it was getting changed!
			self_parent = copy.copy(self_old.parent)
			
			# Knowing if it is a left child or a right child
			if (self_old.ind==self_parent.left.ind):
				isLeft=True
			else:
				isLeft=False
				
			self_temp.prune_step()			
			if isLeft:	
				self_parent.left=self_temp
			else:
				self_parent.right=self_temp
			
			self_new = copy.copy(self_parent.get_updated_root())
	
		## If both the left and right children are leaves
		if (self_old.left.isLeaf!=-1 or (self_old.left.left==None and self_old.left.right==None)) and (self_old.left.isLeaf!=-1 or (self_old.right.left==None and self_old.right.right==None)):	
			return [self_new] 
	
		## If only left child is a leaf
		elif self_old.left.isLeaf!=-1 or (self_old.left.left==None and self_old.left.right==None):
		
			## Bug fix
			self_old.right.parent = self_old
			
			return  (self_old.right.prune_all()) + [self_new] 
			
		## If only right child is leaf	
		elif self_old.left.isLeaf!=-1 or (self_old.right.left==None and self_old.right.right==None):

			## Bug fix
			self_old.left.parent = self_old
		
			return (self_old.left.prune_all()) + [self_new] 

		## If none are leaves
		else:
			### Bug fix
			self_old.left.parent = self_old
			self_old.right.parent = self_old
			
			return (self_old.left.prune_all()) + (self_old.right.prune_all()) + [self_new] 
	'''
'''		
### An independant total accuracy function
def total_accuracy(root, Y, X):
	return root.overall_accuracy(Y,X)
	
### An independant function for Getting the best pruned tree!
def best_prune(tList, Y, X):
	vfunc = np.vectorize(total_accuracy, excluded = ['Y','X'])
	output = vfunc(root = tList,Y = Y,X = X)
	maxAcc = output.max()	
	maxInd = output.argmax()
	print(len(output), maxAcc, output[0])
			
	return tList[maxInd], maxAcc		
'''
							
#### Initiating our tree: 
## (1): Ind, Med is -1 indicating that no variable is conditioned here
## (2): isGrown is False here
## (3): isLeaf is -1 here to indicate that the given node is not a leaf in the true
print("Decision Tree without Pruning started")
start_time = time.time()

root = Node(class_train, features_train, -1, -1, False, -1, None, class_test, features_test, class_valid, features_valid)

## Initiating the plot variables
height_list = []
node_list = []
train_acc_list = []
test_acc_list = []
valid_acc_list = []

### Running the loop depending on the Overall height of the tree
### The maximum height of the tree obtained is 52. Overall Nodes about 20,000! Train accuracy of 90%, Test and Valid accuracies of 77%
for i in range(60):
	print(root.height())
	
#	node_list.append(root.total_nodes())
#	height_list.append(root.height())
#	train_acc_list.append(root.overall_accuracy(class_train,features_train))
#	test_acc_list.append(root.overall_accuracy(class_test,features_test))
#	valid_acc_list.append(root.overall_accuracy(class_valid,features_valid))

	root.insert()
	
print("The required tree has been constructed")
print()

'''
## Total time taken
print("Total time taken is ", time.time() - start_time)

### Plotting them

## Adding the few further elements
node_list.append(20008)
train_acc_list.append(train_acc_list[len(train_acc_list)-1])
test_acc_list.append(test_acc_list[len(test_acc_list)-1])
valid_acc_list.append(valid_acc_list[len(valid_acc_list)-1])

plt.plot(node_list, train_acc_list, label='Train Accuracy')
plt.plot(node_list, test_acc_list, label='Test Accuracy')
plt.plot(node_list, valid_acc_list, label='Validation Accuracy')
plt.legend()
plt.xlabel('Total Number of Nodes')
plt.ylabel('Accuracy')
plt.show()
'''
initial_nodes = root.total_nodes()
new_nodes = initial_nodes

orig_train_acc = root.overall_accuracy(class_train, features_train)
orig_test_acc = root.overall_accuracy(class_test, features_test)
orig_valid_acc = root.overall_accuracy(class_valid, features_valid)

node_list.append(0)
train_acc_list.append(orig_train_acc)
test_acc_list.append(orig_test_acc)
valid_acc_list.append(orig_valid_acc)

#### Pruning the maximum tree. Also limiting the size of the nodes
while(new_nodes>=10000):
	
	'''
	## Storing stuff in the Lists for plotting the graph
	node_list.append(root.total_nodes())
	train_acc_list.append(root.overall_accuracy(class_train,features_train))
	test_acc_list.append(root.overall_accuracy(class_test,features_test))
	valid_acc_list.append(root.overall_accuracy(class_valid,features_valid))
	'''
	
	## Creating a copy of the older root
	root_old = copy.copy(root)
#	tList = root.prune_all()
	
	## Storing the original Validation accuracy
	orig_acc = root_old.overall_accuracy(class_valid, features_valid)
	print(orig_acc, root_old.total_nodes(), root_old.height())
	
	## Best Accuracy
	dump, bestPrune, status = root_old.get_best_prune(orig_acc, class_valid, features_valid)
	bestAcc = bestPrune.overall_accuracy(class_valid, features_valid)
	new_nodes = bestPrune.total_nodes()
	print(new_nodes)
	
	## Getting the accuracies in the list
	if bestAcc>orig_acc:
		node_list.append(initial_nodes - new_nodes)
		train_acc_list.append(bestPrune.overall_accuracy(class_train, features_train))
		test_acc_list.append(bestPrune.overall_accuracy(class_test, features_test))
		valid_acc_list.append(bestAcc)
	
	## Getting the maximum pruned root
#	bestPrune, bestAcc = best_prune(tList,class_valid, features_valid)
	
	## Checking if the pruned trees are better or worse
	if status!=-1:
		break
	else:
		root = copy.copy(bestPrune)


## Total time taken
print("Total time taken is ", time.time() - start_time)

### Plotting them
#plt.plot(node_list, train_acc_list, label='Train Accuracy')
#plt.plot(node_list, test_acc_list, label='Test Accuracy')
plt.plot(node_list, valid_acc_list, label='Validation Accuracy')
plt.legend()
plt.xlabel('Number of Nodes removed')
plt.ylabel('Accuracy')
plt.show()

