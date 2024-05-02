# Loan Default Prediction
Building a model like this is important because in order to asses whether or not to give a loan out to a client, or even what interest rate to charge them, is a complicated process that has historically been riddled with biases and non-consistent standards. Being able to utilize machine learning techniques to assess the credit-worthiness of consumers with an objective look at these features will help make the lending process far more transparent and equitable. We include an Ethical Discussion section as well to address ethical issues that could arise from the usage of this model as lending is a very important economic process that has the power to make the world a lot more unequal as well.
In our exploration, we explored three different approaches to our machine learning algorithm: logistic regression, support vector machine, and decision trees/random forest. After preprocessing the data and encoding the categorical variables (and scaling the quantitative variables), we fitted these three models and compared the results.

## Introduction to data
This dataset on Loan Default Prediction comes from Kaggle and has 255,347 cases and 17 predictor columns. When we one-hot encoded the categorical features we ended up with 24 predictors. None of the variables were highly correlated with each other and there were no outliers. We also standardized the numeric variables. The predictor names are as follows:

**Numeric**
- `Age`
- `Income`
- `LoanAmount`
- `CreditScore`
- `MonthsEmployed`
- `NumCreditLines`
- `LoanTerm`
- `InterestRate`
- `DTIRatio`

**Categorical**
- `Education` (4 levels)
- `EmploymentType` (4 levels)
- `MaritalStatus` (3 levels)
- `HasMortgage` (2 levels)
- `HasDependents` (2 levels)
- `LoanPurpose` (5 levels)
- `HasCoSigner` (2 levels)

One thing we need to look out for is the imbalanced class distribution in the target variable: Only 12% of the cases had a `Default` value of 1 which makes sense given that most of the time people do not default on their loans. Given the high number of cases, we were able to undersample the majority class using `RandomUnderSampler` from imblearn. 

## Model Selection and Tuning

We initially tested 3 different models for our classification: Logistic Regression, Support Vector Machine, and Random Forest, each validated and fine-tuned using cross validation. The stability of our cross-validation scores did not indicate any high variance or overfitting so dimensionality reduction was not a priority. 

Precision and recall are the primary metrics with which we will evaluate the models. In this context, precision refers to the accuracy of positive predictions, the proportion of cases predicted to default on loans who *actually* will, while recall is the proportion of cases that default that are *predicted* to default. It is unclear which metric is more important: Banks may experience more loss from lower recall, as falsely negative predictions means that someone was granted a loan and defaulted. From a humanitarian perspective, though, lower precision means that more people will have their loan applications rejected or face higher rates without a reliable reason. 

### Random Forest
Random forests are powerful ensemble learning methods that combine many decision tree classifiers to improve predictive performance and prevent overfitting. To build our random forest model, we used scikit-learn’s RandomForestClassifier. We performed a grid search with cross-validation over a range of hyperparameters to tune the model. 

The best performing hyperparamters, as determined by grid search, were:

•	n_estimators = 200
•	max_depth = 80
•	min_samples_leaf = 1
•	min_samples_split = 2

With these hyperparamters, the random forest classifier achieved an accuracy of 93.55% on the test set. 

We also generated a classification report and confusion matrix to evaluate the model’s performance. The classification report showed a precision of 89% for the 0 class and a recall of 87% for the positive class (default=1). The confusion matrix provided a visual representation of the true positives, true negatives, false positives, and false negatives.

Random forests have several advantages that make them well-suited for this dataset:

1.	They can automatically handle non-linear relationships and feature interactions. 
2.	They are robust to outliers and noise in the data. 
3.	They provide feature importance scores to understand the most predictive variables. 
4.	They tend to have low bias and are resistant to overfitting through techniques like bagging. 
The main downside is that random forests are black-box models that lack the same level of interpretability as logistic regression does. However, their superior predictive performance and other advantages outweighed this concern. 

### SVM
Support Vector Machines are a particularly useful classification mechanism because our data set has a lot of features. This makes SVM particularly useful because it is effective in high-dimensional spaces. Additionally, we can adjust the kernel type and slack parameter (C) to make the model be the best fit to our data. Using grid search as well as a series of trial-and-error attempts, we arrived at a slack parameter value of C = 0.5 and the kernel type set to rbf. We set the class_weights to 'balanced' but had experimented with adjusting the class weight ratio in the model since our main obstacle was that instances of the positive class (where default=1, or the people did default on their loans) was significantly outnumbered by instances of the negative class in our dataset. Instead of addressing this issue by adjusting the class weights, however, we ended up deciding to use sklearn's RandomUnderSampler - a tool that handles class imbalance by undersampling the majority class. We tried using both RandomUnderSampler and SMOTE (a similar technique that oversamples the minority class rather than undersampling the majority class) but decided to use RandomUnderSampler because it yielded better results in terms of predicting instances of the minority class.

Using all of the predictors,our SVM model yielded the following metrics:
 - precision: 0.21
 - Recall: 0.70
 - Accuracy: 0.66

Because potential users of this model (banks and other lending companies) would care more about predicting true positives that true negatives (in other words, they care more about being able to accurately predict people defaulting on their loans than predicting people to not default), we prioritized recall over precision in making this model




### Logistic Regression
One strength of logistic regression for this data is that its decision function returns a value between 0 and 1 that we can interpret as a probability that someone defaults. Additionally, the logistic regression function's coefficients can be interpreted just like linear regression's, with greater absolute values indicating a greater importance, assuming they are all scaled similarly. There is also the option of using regularization parameters `l1`, `l2`, and `elasticnet` in order to minimize coefficient values, resulting in a more stable model. We did not include these penalties in our model because cross validation showed stable performance, likely due to the large number of data points in our training set and the absence of outliers. 

Using all the predictors, our initial model had the following metrics:
- Precision: 0.22
- Recall: 0.70
- Accuracy: 0.68

We can visualize the distribution of predicted default probabilities to see how the model did:

!["Predicted Probabilities"](writeup_images/log_reg_initial_probas.png)


This plot illustrates how cases where someone did not default (`Actual` value = 0) generally got low probabilites of default, resulting in true negative classifications yet a great proportion of them were past that 0.5 threshold, resulting in false positives. Similarly, the probabilities of default for postitive cases are a little skewed left, allowing a greater recall, that is, a higher proportion of positive default cases being identified.

The calculated precision and recall are based on a decision threshold of 0.5, which means that someone is predicted to default if the model returns a probability greater than 50%. Changing this threshold can change the model's expected precision and recall, depending on what one values more. 

!['Prediction threshold metrics'](writeup_images/prediction_threshold_metrics.png)

We also tried adding second degree polynomial features to the logistic regression, as interactions between variables can reveal new correlations. We then narrowed it down to 40 variables based on feature importance from a regularized lasso regression. Here are the results:

- Precision: 0.36
- Recall: 0.33
- Accuracy: 0.86 

This model's predicted probabilities are slightly more skewed right, meaning that there are fewer false positives but also fewer false negatives. Looking at the coefficients for the two models can also tell us about what features were most important:

!['logistic regression features'](writeup_images/log_reg_coefs.png)

!['polynomial features'](writeup_images/poly_feature_coefs.png)


**Some observations:**
- There is a strong negative correlation between `Age` and `Default`, indicating that older applicants are predicted to be less likely to default on their loans
- Similarly, there is a positive correlation for `Unemployed` (a categorical variable) as well as `InterestRate`, which makes sense in the context of this problem
- The negative coefficient for `Income * LoanAmount` indicates that the extent to which `LoanAmount` is correlated with defaulting somewhat depends on `Income`, making it so a larger income makes someone less likely to default on a larger loan. 


## Comparing Models


Here is a summary chart of the metrics for our various models:

!['Classification metrics for different models chart'](writeup_images/metrics_for_different_models.png)

We can also see the distribution of false and true positives with these confusion matrices:

!["Confusion Matrices"](writeup_images/confusion_matrices.png)


Due to the nature of loan defaults, recall is more important than precision in the lender’s perspective. A higher recall means that the model can accurately identify a higher proportion of potential defaulters. Missing these cases, false negatives, would result in granting loans to applicants who may eventually default, leading to greater financial losses for the lender. 

Looking at the recall values, the random forest, 0.87, and logistic regression, 0.70, models perform reasonably well in identifying true positive cases of loan defaults. However, the logistic regression model has a lower precision of 0.22, indicating a higher number of false negatives (4,281) compared to the other models. This aligns with its high precision and recall scores. 

The polynomial logistic regression model shows an improvement over the initial logistic regression model, with higher precision, 0.36, and accuracy, 0.86, but still lower recall, 0.33, than the random forest. 

The SVM model has the lowest precision (0.21) and recall (0.70) among the evaluated models, making it the least suitable for this task. 

Based on this, the random forest model is the most appropriate choice for this loan default prediction task, as it strikes the best balance between precision and recall, with a clear emphasis on maximizing recall to minimize financial losses for the lender. 

While the logistic regression models provide more interpretability through their feature importance plots, the superior predictive performance of the random forest outweighs this advantage, in the context of the lender’s priority of accurately identifying potential defaulters. 

## Ethical discussion
Our machine learning model to predict whether or not individuals will default on their loans, and all those like it, has the potential to make lending a much more equitable and transparent process. The usage of models like ours has the potential to enhance the efficiency and fairness of financial services. However, we would be remiss if we did not highlight its potential to be used unethically. It is imperative to ensure that such powerful tools are used responsibly and fairly. 

Improper usage of this model could lead to further discrimination and lack of accountability in the commercial lending process. Lending has historically been very unfair to people of certain racial groups and socioeconomic statuses and because these patterns are reflected in training data, banks and lending organizations who come to use models like these to make lending decisions need to be cautious of that. Regarding lack of accountability, without proper oversight, decisions made by the model could be inexplicable and irreversible, leaving individuals adversely affected by the model with no recourse. It is extremely important to maintain a high level of transparency about how the model operates. Financial institutions should be able to explain the factors leading to any particular decision, and individuals should have access to explanations of decisions that affect them. Unlike neural networks, the models we used here can output feature importance values, leading to a higher degree of transparency yet somewhat lack in classification ability. 

Predicting loan default is a very important decision because being able to receive or not receive a loan has tremendous implications for one’s financial wellbeing. Machine learning models have the ability to expedite the approval process and could be said to prevent human biases yet may end up reinforcing historical biases to the detriment of society at large. Because these decisions cannot afford to be taken lightly, we should strive to further research how we can integrate machine learning into the decision making process while still ensuring its fairness. 

## Conclusion
While none of our models are necessarily strong enough to implement in a real world setting, they did illustrate some of the complexities of modeling real world situations, especially ones with high stakes. The variation in models are great examples of the precision-recall tradeoff, especially with such imbalanced classes. In some ways, these algorithms can promote fairness and decrease human bias but are still limted by the nuances of life experiences. 
