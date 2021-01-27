### Importing and Setting the basic Modules
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import time

# For removing stop words
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 

# For stemming 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()

# For random number generation
from random import seed
from random import randint
seed(1)

# Importing NB and Tokenizer modules
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2

# Importing the ROC metrics
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

### Basic functions for the data processing

# Function to split the string as per space characters
def returns_list(line):
  multi_dim_list = line.split()
  return multi_dim_list

# Function to remove the special characters in a string 
def rem_spl(strin):
	alphanumeric = ""
	for character in strin:
		if character.isalnum():
			alphanumeric += character
	return alphanumeric

# Function to remove the stop words from a token list
def rem_stop(word_tokens):
	filtered_sentence = [] 	  
	for w in word_tokens: 
		if w not in stop_words: 
			filtered_sentence.append(w) 	
	return filtered_sentence	  

# Function to perform stemming
def obt_stem(word_tokens):
	filtered_sentence = []
	for w in word_tokens:
		filtered_sentence.append(ps.stem(w))
	return filtered_sentence

# Function to remove the twitter handles
def rem_twitter(word_tokens):
	filtered_sentence = []
	for w in word_tokens:
		if (w[0]!='@'):
			filtered_sentence.append(w)
	return filtered_sentence
	
# The combined overall function of data processing for Part(D)
def data_proc(word_tokens):
	filtered_sentence = [] 	  
	for w in word_tokens:
		# Checking for the twitter handle 
		if (w[0]=='@'):	
			continue
		else:
			# Removing the special characters in the word 
			w = rem_spl(w)
			
			# If the processed word is empty or just a letter, then it means that it was just a punctuation. So can be removed
			if w=="" or len(w)==1:
				continue			
			else:
				# Checking for stop words now
				if w not in stop_words:
					# Carrying out stemming
					filtered_sentence.append(ps.stem(w))
					
	return filtered_sentence	  


######################################################################################################################################################################################################

### Other Important functions

## Function to read the CSV training files: 0 -> Without processing, 1 -> With processing
def read_csv(val):
	polarity_train = []
	text_train = []
	if val==0:
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/training.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue
				polarity_train.append(row[0])
				text_train.append(returns_list(row[5]))
	
	elif val==1:
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/training.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue			
				polarity_train.append(row[0])
				text_train.append(data_proc(returns_list(row[5])))
	
	else: 
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/training.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue			
				polarity_train.append(row[0])
				text_train.append(row[5])
	
				
	return polarity_train, text_train

## Function to read the CSV test files: 0 -> Without processing, 1 -> With processing
def read_csv_test(val):
	polarity_train = []
	text_train = []
	if val==0:
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/test.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue
				polarity_train.append(row[0])
				text_train.append(returns_list(row[5]))
	
	elif val==1:
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/test.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue			
				polarity_train.append(row[0])
				text_train.append(data_proc(returns_list(row[5])))
				
	else:
		# Using CSV functions to read the values row-wise
		with open("/home/vedant/Downloads/COL774/Ass2/trainingandtestdata/test.csv", 'r',encoding='latin-1') as f:
			reader = csv.reader(f)
			for row in reader:
				if (row[0]=='2'):
					continue			
				polarity_train.append(row[0])
				text_train.append(row[5])
	
				
	return polarity_train, text_train

## Defining the dictionary to store the string tokens
class my_dictionary(dict):
    
    def __init__(self):
        self = dict()
        
    def add(self, key, value):
        self[key] = value
    
    def checkKey(self, key):
        if key in self.keys():
            return True
        else:
            return False

## Function to get the value of phi_y
def get_phi_y(polarity_train,text_train):
	sumVal = 0.
	m = len(polarity_train)
	for i in range(m):
		if (polarity_train[i] == 4):
			sumVal = sumVal+1.
	return (sumVal/m)

## Function to get the denominator value of phi_k given Y=y (Either 0 or 4)
def get_cond(polarity_train,text_train):
	sumDenom0 = 0.
	sumDenom1 = 0.	
	m = len(polarity_train)	
	for i in range(m):
		if polarity_train[i] ==0:
			sumDenom0 = sumDenom0 + (len(text_train[i]))
		else:
			sumDenom1 = sumDenom1 + (len(text_train[i]))
		
	return sumDenom0, sumDenom1

## Function to get the prob of Y=y|X
def get_pred(polarity_train,text_train,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V):
	m = len(polarity_train)
	out_array = []
	for i in range(m):
		sumVal_0 = 0.
		sumVal_4 = 0.
		ni = len(text_train[i])
		for j in range(ni):
		
			if not dict_obj0.checkKey(text_train[i][j]):
				dict_obj0.add(text_train[i][j],1./(sumDenom0 + V))
				
			if not dict_obj1.checkKey(text_train[i][j]):
				dict_obj1.add(text_train[i][j],1./(sumDenom1 + V))
							
			sumVal_0 = sumVal_0 + np.log(dict_obj0[text_train[i][j]])
			sumVal_4 = sumVal_4 + np.log(dict_obj1[text_train[i][j]])
		temp = 0
		if (sumVal_4 > sumVal_0):
			temp =4
			
		tempBool = True
		if (temp != polarity_train[i]):
			tempBool=False
			
		out_array.append([polarity_train[i],tempBool])
		
	return out_array
	
## Function to get the accuracy as per the probabilities
def get_accuracy(in_list):
	m = len(in_list)
	sumVal = 0.
	for i in range(m):
		if (in_list[i][1]):
			sumVal = sumVal + 1
	return (sumVal/m)

## Function to get the accuracy as per random guessing
def get_accuracy_random(polarity_test):
	sumVal = 0.
	m = len(polarity_test)
	
	for i in range(m):
		val = 4 * randint(0,1)
		if (val==polarity_test[i]):
			sumVal = sumVal+1
	return sumVal/m

## Function to get the accuracy according to majority prediction
def get_accuracy_majority(polarity_test,majority):
	sumVal=0.
	m = len(polarity_test)
	
	for i in range(m):
		if (polarity_test[i] == majority):
			sumVal = sumVal + 1
	return sumVal/m

## Function to get the confusion matrix
def get_confused(in_list):
	out_mat = [[0,0],[0,0]]
	m = len(in_list)
	for i in range(m):
		# True positives:0
		if in_list[i][1] and (in_list[i][0] ==0):
			out_mat[0][0] = out_mat[0][0]+1
		# True Negatives:4
		elif in_list[i][1] and (in_list[i][0] ==4):
			out_mat[1][1] = out_mat[1][1] + 1
		# False positives
		elif (not in_list[i][1]) and (in_list[i][0] ==0):
			out_mat[1][0] = out_mat[1][0] + 1
		# False Negatives
		else:
			out_mat[0][1] = out_mat[0][1] + 1
	return out_mat
	
##################################################################################################################################################################################################

'''
### Part(A): Basic Multinomial Model implementation of Naive Bayes
print("Part A started")

## Reading the CSV values
polarity_train, text_train = read_csv(0)
polarity_test, text_test = read_csv_test(0)

# Conersion from string to Int 
polarity_train = np.asarray(polarity_train).astype(np.int)
polarity_test = np.asarray(polarity_test).astype(np.int)

# Getting the dictionaries for the 0 and 4 case 
dict_obj0 = my_dictionary()
dict_obj1 = my_dictionary()

m = len(polarity_train)
V = 0
# Storing the values in the dictionary
for i in range(m):
	x = text_train[i]
	ni = len(x)
	for j in range(ni):
		
		x = text_train[i][j]
				
		if polarity_train[i] ==0:
			if dict_obj0.checkKey(x):
				dict_obj0[x] = dict_obj0[x] + 1.
			else:
				dict_obj0.add(x,1.)
				V=V+1
		else:
			if dict_obj1.checkKey(x):
				dict_obj1[x] = dict_obj1[x] + 1.
			else:
				dict_obj1.add(x,1.)
				V=V+1

print('Done')
# Getting the Denomiator values
sumDenom0, sumDenom1 = get_cond(polarity_train,text_train)

# Getting probabilities in the Dictionary values
for key in dict_obj0.keys():
	dict_obj0[key] = (dict_obj0[key]+1.)/(sumDenom0+V)

for key in dict_obj1.keys():
	dict_obj1[key] = (dict_obj1[key]+1.)/(sumDenom1+V)

## Getting the Accuracy for the Training and Testing file
acc_train = get_accuracy(get_pred(polarity_train,text_train,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
acc_test = get_accuracy(get_pred(polarity_test,text_test,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
print(acc_train,acc_test)

### Part (B): Random/Majority baseline

print()
print("Part B started")
print(get_accuracy_random(polarity_test), get_accuracy_majority(polarity_test,1))

### Part (C): Confusion matrix
print()
print("Part C started")
print(get_confused(get_pred(polarity_test,text_test,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V)))
'''

###################################################################################################################################################################################################

'''
### Part (D) : Transformed Input
print()
print("Part D started")


## Reading the CSV values
polarity_train, text_train = read_csv(1)
polarity_test, text_test = read_csv_test(1)

polarity_train = np.asarray(polarity_train).astype(np.int)
polarity_test = np.asarray(polarity_test).astype(np.int)


# Getting the dictionaries for the 0 and 4 case 
dict_obj0 = my_dictionary()
dict_obj1 = my_dictionary()

m = len(polarity_train)
V = 0
# Storing the values in the dictionary
for i in range(m):
	x = text_train[i]
	ni = len(x)
	for j in range(ni):
		
		x = text_train[i][j]
				
		if polarity_train[i] ==0:
			if dict_obj0.checkKey(x):
				dict_obj0[x] = dict_obj0[x] + 1.
			else:
				dict_obj0.add(x,1.)
				V=V+1
		else:
			if dict_obj1.checkKey(x):
				dict_obj1[x] = dict_obj1[x] + 1.
			else:
				dict_obj1.add(x,1.)
				V=V+1

print('Done')
# Getting the Denomiator values
sumDenom0, sumDenom1 = get_cond(polarity_train,text_train)

# Getting probabilities in the Dictionary values
for key in dict_obj0.keys():
	dict_obj0[key] = (dict_obj0[key]+1.)/(sumDenom0+V)

for key in dict_obj1.keys():
	dict_obj1[key] = (dict_obj1[key]+1.)/(sumDenom1+V)

## Getting the Accuracy for the Training and Testing file
acc_train = get_accuracy(get_pred(polarity_train,text_train,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
acc_test = get_accuracy(get_pred(polarity_test,text_test,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
print(acc_train,acc_test)
'''

#####################################################################################################################################################################################################

### Part (E): Feature Engineering
## Here, we are using the same training and testing lists as obtained from the Part D above 

'''
## Feature 1: Making the use of Bigrams
print()
print("Bigram formation started")

# Creating Bigrams
text_train_E_1 = []
for p in range(len(text_train)):
	res = []
	for j in range(len(text_train[p])-1):
		res.append(text_train[p][j] + '-' + text_train[p][j+1])
	text_train_E_1.append(res)

text_test_E_1 = []
for p in range(len(text_test)):
	res = []
	for j in range(len(text_test[p])-1):
		res.append(text_test[p][j] + '-' + text_test[p][j+1])
	text_test_E_1.append(res)

# Getting the dictionaries for the 0 and 4 case 
dict_obj0 = my_dictionary()
dict_obj1 = my_dictionary()

# Storing the values in the dictionary
m = len(polarity_train)
V = 0
for i in range(m):
	x = text_train_E_1[i]
	ni = len(x)
	for j in range(ni):
		
		x = text_train_E_1[i][j]
				
		if polarity_train[i] ==0:
			if dict_obj0.checkKey(x):
				dict_obj0[x] = dict_obj0[x] + 1.
			else:
				dict_obj0.add(x,1.)
				V=V+1
		else:
			if dict_obj1.checkKey(x):
				dict_obj1[x] = dict_obj1[x] + 1.
			else:
				dict_obj1.add(x,1.)
				V=V+1

print('Done')
# Getting the Denomiator values
sumDenom0, sumDenom1 = get_cond(polarity_train,text_train_E_1)

# Getting probabilities in the Dictionary values
for key in dict_obj0.keys():
	dict_obj0[key] = (dict_obj0[key]+1.)/(sumDenom0+V)

for key in dict_obj1.keys():
	dict_obj1[key] = (dict_obj1[key]+1.)/(sumDenom1+V)

## Getting the Accuracy for the Training and Testing file
acc_train = get_accuracy(get_pred(polarity_train,text_train_E_1,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
acc_test = get_accuracy(get_pred(polarity_test,text_test_E_1,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
print(acc_train,acc_test)


## Feature 2: Making the use of skipgrams

print()
print("Skipgram formation started")

# Creating Skipgrams
text_train_E_2 = []
for p in range(len(text_train)):
	res = []
	for j in range(len(text_train[p])-1):
		for k in range(j+1,len(text_train[p])):
			res.append(text_train[p][j] + '-' + text_train[p][k])
	text_train_E_2.append(res)

text_test_E_2 = []
for p in range(len(text_test)):
	res = []
	for j in range(len(text_test[p])-1):
		for k in range(j+1,len(text_test[p])):
			res.append(text_test[p][j] + '-' + text_test[p][k])
	text_test_E_2.append(res)

# Getting the dictionaries for the 0 and 4 case 
dict_obj0 = my_dictionary()
dict_obj1 = my_dictionary()

# Storing the values in the dictionary
m = len(polarity_train)
V = 0
for i in range(m):
	x = text_train_E_2[i]
	ni = len(x)
	for j in range(ni):
		
		x = text_train_E_2[i][j]
				
		if polarity_train[i] ==0:
			if dict_obj0.checkKey(x):
				dict_obj0[x] = dict_obj0[x] + 1.
			else:
				dict_obj0.add(x,1.)
				V=V+1
		else:
			if dict_obj1.checkKey(x):
				dict_obj1[x] = dict_obj1[x] + 1.
			else:
				dict_obj1.add(x,1.)
				V=V+1

print('Done')
# Getting the Denomiator values
sumDenom0, sumDenom1 = get_cond(polarity_train,text_train_E_2)

# Getting probabilities in the Dictionary values
for key in dict_obj0.keys():
	dict_obj0[key] = (dict_obj0[key]+1.)/(sumDenom0+V)

for key in dict_obj1.keys():
	dict_obj1[key] = (dict_obj1[key]+1.)/(sumDenom1+V)

## Getting the Accuracy for the Training and Testing file
acc_train = get_accuracy(get_pred(polarity_train,text_train_E_2,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
acc_test = get_accuracy(get_pred(polarity_test,text_test_E_2,dict_obj0,dict_obj1,sumDenom0,sumDenom1,V))
print(acc_train,acc_test)

'''

######################################################################################################################################################################################################


#### Part(F): TF-IDF
print()
print("TF-IDF started")
start_time = time.time()

## Reading the features and labels as per th;e Data processing done before
polarity_train, text_train = read_csv(2)
polarity_test, text_test = read_csv_test(2)

polarity_train = np.asarray(polarity_train).astype(np.int)
polarity_test = np.asarray(polarity_test).astype(np.int)

vectorizer = TfidfVectorizer(min_df=0.001)
#vectorizer = TfidfVectorizer()
clf = GaussianNB()

# Doing "fit" and "transform" differenly to fit the test data to the training features i.e the same dictionary
temp_fit = vectorizer.fit(text_train)
transfomed_train = vectorizer.transform(text_train)

# Doing the Percentile term 
#sp = SelectPercentile(chi2, percentile=0.1)
#transfomed_train = sp.fit_transform(transfomed_train,polarity_train)

transfomed_test = vectorizer.transform(text_test)
#transfomed_test = sp.transform(transfomed_test)

# Having a batch size
batch = 1000

# Partial Fitting the GaussianNB Model
for i in range(len(polarity_train)//batch):
	print(i)
	clf.partial_fit(transfomed_train[i * batch : (i+1)* batch].toarray(), polarity_train[i * batch : (i+1) * batch], [0,4])
	
# Predict the Values by the model
predicted_test = clf.predict(transfomed_test.toarray())

# Printing the score of the model
print(sklearn.metrics.accuracy_score(polarity_test, predicted_test))

fpr,tpr,_ = roc_curve(polarity_test, clf.predict_proba(transfomed_test.toarray())[:,1], pos_label = 4)

plt.figure()
plt.plot(fpr,tpr, label='Naive Bayes')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.savefig('ROC_curve.png')
plt.show()

# Printing the total time taken in the processed
print("The total time taken: ", time.time() - start_time)



