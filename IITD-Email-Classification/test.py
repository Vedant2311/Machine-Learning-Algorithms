### Importing and Setting the basic Modules
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

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
	with open(file_path,'r', encoding='latin-1') as fp:
		reader = csv.reader(fp)
		count = 0
		for row in reader:
			# Ignoring the first line of the training file
			if count==0:
				count=1
				continue
			subject_train.append(preprocess_string(row[0],'subject'))
			content_train.append(preprocess_string(row[1],'content'))
	return subject_train, content_train
	
## Function to writer to a txt file 
def output_writer(class_predicted, output_name):
	with open(output_name, 'w') as fp:
		for values in class_predicted:
			fp.write(str(values)+'\n')

### Defining the main function 
def main():
	
	# Taking command line arguments 
	saved_model = sys.argv[1]
	file_path = sys.argv[2]
	output_name = sys.argv[3]
	subject_test, content_test = csv_reader(file_path)
	
	# Loading the vectorizer
	vectorizer = pickle.load(open('vectorizer.pkl','rb'))
	
	## Defining a hyper parameter giving the importance of Subject wrt content
	imp_subject = 2
	text_test = [content + (imp_subject)*subject for (content, subject) in zip(content_test, subject_test)]	
		
	# Doing "fit" and "transform" differenly to fit the test data to the training features i.e the same dictionary
	transformed_test = vectorizer.transform(text_test)
	
	# Load the saved model and predict the values
	loaded_model = pickle.load(open(saved_model, 'rb'))
	class_predicted = loaded_model.predict(transformed_test.toarray())
	output_writer(class_predicted, output_name)		
	
## Calling the main function
if __name__ == "__main__":
	main()
