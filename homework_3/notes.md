critical_temp is the target variable
(300, 82) - size of the dataset, perfect example for regularization

L2 - Closed form solution with included intercept

L1 - Optimization done using the Powell method

Preprocessing:
- First 200 rows are for training, the rest are used for testing
- RMSE used as the evaluation metric
- First X_training and y_training are standardized, then X_test and y_test are standardized using the parameters of the training data (very important!!!)