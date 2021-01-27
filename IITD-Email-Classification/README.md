# IITD-Email-Classification-System

Consists of a thoroughly engineered Test-categorization-system, that would serve as a potential candidate to be used for the Email classification system employed by IIT-Delhi. Was among one of the best performants in the class tournament.

## Problem-statement
The detailed problem statement, along with the specific usage for the program can be found in the document **Problem.pdf**. On a broad scale, it was required that we make use of the Email content (and possibly the Email subject) as the training input and come up with a system that classifies that into one of the seven different classes, with a high accuracy. The scripts were written in python.

## Experimentations
### Feature Engineering
Different feature engineering methods and different combinations of these were taken into account.
```
* Lemmatizing a word if it is not all-caps
* Removing stop-words and Encrypted-words
* Replacing long https links by a <LINK> token
* Replacing numbers by a <NUM> token
* Adding bi-grams and skip-grams
* Adding a special token for the mail subject 
* Adding the subject tokens twice to the input features (up-weighting)
* Performing Tf-Idf vectorization with a sublinear Term Frequency
* POS-tagging using nltk modules
```

### Different models considered
The following different non-neural models were considered here. Also, a proper hyper-parameter tuning was done for each method based on the given validation set as well as by having K-fold validation done from the train data.
```
* Linear Regression
* Logistic Regression
* Gaussian Naive Bayes
* K-neighbours classifier
* Random Forests
* Decision Trees
* Support Vector Machines
```

