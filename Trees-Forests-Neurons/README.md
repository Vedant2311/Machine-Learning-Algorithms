# Trees-Forests-Neurons

Decision-Trees, Random-Forests, and Neural-Networks are implemented **from scratch** using python. 

The problem statement for the analysis and reasoning regarding these different methods can be found as **Ass3_fall20.pdf** (Note that this is somewhat different from the actual problem statement on which the system was implemented, but apart from some minor modifications they are the same). All the explanations and analysis can be found in the **Report.pdf** file, present in the different directories here.

## Forests-of-Trees
The first section here deals with construction of a Decision Tree using the data obtained from the VirusShare dataset. At each node, the attribute is selected that leads to the maximum decrease in the entropy of the class variable. The accuracies obtained have been plotted against the size of the tree as the tree grows and subsequently proper analysis has been made.

And as could be noticed from the above plot, there is a problem of over-fitting with this algorithm and for that a post-pruning based on the validation set was applied. And the results obtained along with the pruning were plotted as well.

Then finally, Random forest was implemented using the Scikit-learn library and different forests were grown according to the different model hyper-parameters and a grid-search was performed in order to get the best set of parameters. Proper parameter sensitivity analysis was also carried out.

All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

## Alphabets-Detection-Library
The second section here is based on the creating a neural network in order to classify the images of different alphabets to their actual value. A generic neural network architecture as built, that would have the hyper-parameters as the *Mini-batch size*, *Number of features*, *Number of target classes*, and *Hidden-layer architecture*, *activation function* etc.

Different experimentations were carried out by making use of different activation functions, different hidden layer architectures, constant and adaptive learning rate etc. and there performance (in terms of speed and accuracies obtained) were documented. Finally, the MLPC classifier from the scikit-learn library was used in the same problem situation and the implemented developed from the scratch was compared with that.

All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.
