### Importing and Setting the basic Modules
import time 
begin = time.time()
import numpy as np
import csv
import sys
import re
csv.field_size_limit(sys.maxsize)

# Importing nltk libraries
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.util import ngrams
from nltk.util import skipgrams

# Importing scikit-learn libraries
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

## Function to convert the string into interger type
def get_type(str_in):
	if str_in=="Meeting and appointment":
		return 1
	elif str_in=="For circulation":
		return 2
	elif str_in=="Selection committe issues":
		return 3
	elif str_in=="Policy clarification/setting":
		return 4
	elif str_in=="Recruitment related":
		return 5
	elif str_in=="Assessment related":
		return 6
	elif str_in=="Other":
		return 7
	else:
		print("Error for: " + str_in)
		return -1
		
## Function to preprocess individual words
def preprocess_word(word):
	# No preprocessing done if the word is all caps
	if word.isupper():
		return word
	# Doing Stemming/lemmatization otherwise
	else:
		return lemmatizer.lemmatize(word.lower())		
		
## Function to preprocess the content/subject string 
def preprocess_string(str_in, str_type):

	# Getting tokens from the sentence
	tokens_init = word_tokenize(str_in)

	# Performing Lemmatization and removing stop-words and encryped-words. Also removing links. Removes special characters from the word. Removes words with one or less chars
	tokens_cleaned = [preprocess_word(re.sub(r'\W+', '', word)) for word in tokens_init if ((word.lower() not in stop_words) and (len(word)<=25))]
#	tokens_cleaned1 = [token for token in tokens_cleaned if ((len(token)>1 or (len(token)==1 and token.isdigit())) and (token != 'http'))]
	tokens_cleaned1=[]
	for token in tokens_cleaned:	
		if len(token)==0:
			continue
		elif token.isdigit():
			tokens_cleaned1.append('NUM')
		elif token=='http':
			tokens_cleaned1.append('LINK')
		elif len(token)>1:
			tokens_cleaned1.append(token)
	
	# Adding skipgrams to the strings
	skip_tokens = list(skipgrams(tokens_cleaned1, 2, 2))
	tokens_skip = [ai + "-" + bi for (ai,bi) in skip_tokens]
	
	# Adding POS tags for each word 
	temp_list = []
	for word in tokens_cleaned1:
		temp_list = temp_list + nltk.pos_tag([word])
	tokens_pos = [ai + "-" + bi for (ai,bi) in temp_list]
	
	# Adding ngrams to the strings
	n_tokens = list(ngrams(tokens_cleaned1,2))
	tokens_ngram = [ai + "-" + ci for (ai,ci) in n_tokens]
	
	# Add the skipgram tokens to the initial ones
	if str_type=='subject':
		tokens_final = tokens_pos + tokens_ngram + tokens_skip
		tokens_final = ["IMP:" + token for token in tokens_final]
		final_str = ' '.join(tokens_final)
		return final_str
	else:
		tokens_final = tokens_pos + tokens_ngram + tokens_skip
		final_str = ' '.join(tokens_final)
		return final_str
		
					
## Function to read the csv files
def csv_reader(file_path):
	subject_train = []
	content_train = []
	class_train = []
	with open(file_path,'r', encoding='latin-1') as fp:
		reader = csv.reader(fp)
		count = 0
		for row in reader:
			# Ignoring the first line of the training file
			if count==0:
				count=1
				continue
			subject_train.append(preprocess_string(row[0], 'subject'))
			content_train.append(preprocess_string(row[1], 'content'))
			class_train.append(get_type(row[2]))
	return subject_train, content_train, class_train
	
### Defining the main function 
def main():
	
	# Taking command line arguments 
	file_path = sys.argv[1]
	to_save_name = sys.argv[2]
	subject_train, content_train, class_train = csv_reader(file_path)
	
	### Converting these lists into float arrays
	class_train = np.asarray(class_train).astype(np.int)
	
	# Defining a vectorizer and getting rid of some terms with minimum Document frequency
	vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf = True)
	
	## Defining a hyper parameter giving the importance of Subject wrt content
	imp_subject = 2
	text_train = [content + (imp_subject)*subject for (content, subject) in zip(content_train, subject_train)]	
		
	# Doing "fit" and "transform" differenly to fit the test data to the training features i.e the same dictionary
	transformed_train = vectorizer.fit_transform(text_train)
	
	# Saving this vectorizer as a pickle file
	pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
	
	### Defining the classifier and training it 
	#Failed models below
	#clf = GaussianNB()
	#clf = KNeighborsClassifier(n_neighbors = 7)
	#clf = RandomForestClassifier()
	#clf = SVC(kernel = 'linear', C = 5, class_weight = 'balanced')
	clf = LogisticRegression(class_weight='balanced', C=2.1, max_iter=1000)
	clf.fit(transformed_train.toarray(), class_train)	
	pickle.dump(clf, open(to_save_name, 'wb'))
	
	'''
	## Beginning of the interfence
	subject_dev, content_dev, class_dev = csv_reader('val.csv')
	class_dev = np.asarray(class_dev).astype(np.int)

	## Defining a hyper parameter giving the importance of Subject wrt content
	imp_subject = 2
	text_dev = [content + (imp_subject)*subject for (content, subject) in zip(content_dev, subject_dev)]	
		
	# Doing "fit" and "transform" differenly to fit the test data to the training features i.e the same dictionary
	transformed_dev = vectorizer.transform(text_dev)
	
	class_predicted = clf.predict(transformed_dev.toarray())
	print(len(list(class_predicted)))
	print(class_predicted)
	print("The scores for val are printed as: Micro-accuracy, Macro-accuracy")
	print(accuracy_score(class_dev, class_predicted),recall_score(class_dev, class_predicted, average='macro'))
	print()	

	class_predicted = clf.predict(transformed_train.toarray())
	print("The scores for train are printed as: Micro-accuracy, Macro-accuracy")
	print(accuracy_score(class_train, class_predicted),recall_score(class_train, class_predicted, average='macro'))
	print()
	
	print("The scores for Cross-validation are printed as: Micro-accuracy, Macro-accuracy")
	print(cross_val_score(clf, transformed_train.toarray(), class_train, cv=5, scoring='accuracy'))
	print(cross_val_score(clf, transformed_train.toarray(), class_train, cv=5, scoring='recall_macro'))
	print()
	'''
	
	# Ending time inidicated here
	end = time.time()
	print("The time taken in minutes is: " + str((end-begin)*1./60.0))
	
## Calling the main function
if __name__ == "__main__":
	main()
