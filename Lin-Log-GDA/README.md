# Lin-Log-GDA
The standard algorithms of Linear-Regression (using both Batch-Gradient-Descent and Stochastic-Gradient-Descent) methods, Logistic-Regression, and Gaussian-Discriminant-Analysis are implemented here using Python. 

The problem statement for the analysis and reasoning regarding these different methods can be found as **Ass1_fall20.pdf** (Note that this is somewhat different from the actual problem statement on which the system was implemented, but apart from some minor modifications they are the same). All the explanations and analysis can be found in the **Report.pdf** file.

## Linear-Regression
The First section here deals with Implementing Batch Gradient Descent for optimising the Linear-Regression loss. The dataset used here corresponds to the Acidicy of the wine and it's density, and the algorithm would learn the relationship between them. All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

## Sampling-And-Stochastic-Gradient-Descent
The Second section here works with the idea of sampling by adding Gaussian noise to the prediction of a hypothesis and thus results in the generation of synthetic training data. And corresponding to this artificial training data, SGD method is implemented with different batch sizes *r* and their performance and convergence are analysed. All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

## Logistic-Regression
The third section involves implementing Newton's method for optimising binary classification noise and then applying it to fit a logistic regression model. All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

## Gausssian-Discriminant-Analysis
The fourth (and the last) section here works with Implementing the GDA algorithm for the Binary classification problem of seperating out Salmons from Alaska and Canada, with the appropriate dataset provided for that. The Linear and Non-Linear assumptions for the boundary condition for this method were considered and were analysed in detail. All the files for this can be found in the respective directory. Kindly go through the code for a better understanding.

