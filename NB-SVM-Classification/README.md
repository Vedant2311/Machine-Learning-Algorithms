# NB-SVM-Classification

The algorithms of Naive-Bayes (for binary classification) and Support-Vector-Machines (for binary as well as multi-class classification) are implemented **from scratch** using Python.

The problem statement for the analysis and reasoning regarding these different methods can be found as **Ass2_fall20.pdf** (Note that this is somewhat different from the actual problem statement on which the system was implemented, but apart from some minor modifications they are the same). All the explanations and analysis can be found in the **Report.pdf** file.

## Yelp-Review-Classifier
The first section here deals with making use of the Naive-Bayes algorithm for classifying the user reviews as Positive and Negative. *Confusion Matrix* and *ROC curve* were created and different Engineering optimisations and Feature-Engineering techniques were applied in-order to get the maximum accuracy. All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

## FMNIST-Article-Classifier
The second section here deals with making use of the Support Vector Machines (SVMs) in order to build an Image classifier. The SVM Optimisation problem was solved using the **CVXOPT** package. Here both the Linear and Gaussian Kernels were considered and their performances were compared and analysed. 

These Binary classifiers were also extended to Multi-class classification by having an ensemble of (<sup>k</sup>C<sub>2</sub>), where *k* would be the number of classes. This Multi-class classifier was also implemented using the Scikit SVM library. And finally **k-fold cross validation** was performed to estimate the best value of the SVM model parameters.

All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.
