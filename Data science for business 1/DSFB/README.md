**Description**

Assignment 1 for Data Science for Business 2 – Predictive Modeling and a Dash of Time Series

**Prerequisites**

- Download the dataset heinz-sales.txt" 

**Task 1. Autoregressive models for weekly sales**

Load the dataset “heinz-sales.txt” from the course website, which contains weekly sales of Heinz ketchup measured in US dollars. In total, we have 123 weekly observations from one supermarket. The dataset is in comma-separated form and contains the following variables:

1. sales (weekly)
2. price (used in the given supermarket)
3. CP – coupon promotion only
4. DP – major display promotion only
5. TP – combined promotion
6. mail campaigns at 4 different sites (count variables)

 - (a) (1pt) Compute descriptive statistics for the data. Use e.g. matplotlib package in Python for plotting histograms and any other charts you deem necessary. Discuss the results briefly.

 - (b) (2pt) Build a simple AR(p) model for sales with a different number of lags p 􏰀 1,2. It is recommended to do the data engineering programmatically.

 - (c) (3pt) Simulate trajectories from the models that you have estimated. You can use the same principle as show in the tutorial and assume a normal distribution for the error terms. Compare the simulated trajectories to the original sales data and see if they are statistically different to see if your implementation is correct.
(Note: This simulation exercise will help you to understand the concepts introduced during Week 2.)
     
**Task 2. Modeling sales and promotion effectiveness**

 - (d) (1pt) Extend your dataset by adding lags for the other variables in addition to the target variable. Consider using 2 lags.
 
 - (e) (1pt) Divide the data into training and testing sets and justify your division scheme. Take into account the time-series nature of the data.

- (f) (2pt) Use regularized regressions (e.g., Lasso, Ridge, and Elastic Net) to build a sparse model for weekly sales. Experiment with 2 models and examine the regularization paths. Based on the information criteria, which value for the parameter lambda is the best? Can we implement the usual cross-validation approach on the model? Explain why or why not.

- (g) (2pt) Choose the model from the step above that you think is best and explain your choice. Analyze the final model. You can re-estimate the model using OLS and compare the coefficients side-by-side with the regularization coefficients. Are there differences? Which set of coefficients would likely give better results on the testing dataset? Compare the models on the testing dataset. Discuss which variables should be selected and how effective promotion campaigns are driving weekly sales.

- (h) (3pt) Suppose you are managing the company and are deciding which campaigns to run for the next 4 weeks (i.e., you have the ability to control the values given to the exogenous variables, including the price). Decide on a suitable campaign and use the model selected in the previous step to simulate sales for the next month (similarly to how you simulated the trajectories in part (c)).

*Note: HTML links don't work properly because the Notebook file converts to a static HTML file*
