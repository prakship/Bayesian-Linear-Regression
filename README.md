### Carried out Model selection in Bayesian Linear Regression from scratch

- Defined functions to evaluate the optimal solution of posterior distribution.
- Next defined functions to evaluate Mean-square-error(MSE), which is used as the measure of accuracy of different models while varying the regularizing parameter.
- Firstly, implemented model selection by varying the regularizing parameter(lambda) and used MSE as the measure to determine the best model.
- Second, implemented 10-fold cross validation to find the best model using MSE measure of model performance.
- Finally, implemented Bayesian model selection using evidence approximation and determined the best model.
- Compared all three methods of model selection with respect to their run times and MSE.

### Results

- Cross validation method provided the best results in terms of accuracy and run time of algorithm.
- However, we should be careful while concluding this result as the runtimes aren't significantly varying and we might have to make compromises for datasets with large sizes.
