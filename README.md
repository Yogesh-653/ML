This projects provides the investigation to the performance of various classification algorithms on the dataset of direct marketing campaigns conducted by a Portuguese banking institution, with the aim of promoting term deposit subscription through phone calls. In this paper we deploy four classification algorithm Nearest Neighbors (KNN), Logistic Regression, Random Forest, and Support Vector Machine (SVM). We systematically tune the hyperparameter of each algorithm using appropriate techniques such as grid search and cross validation to optimize their predictive model. By analyzing the overall techniques of algorithm that achieves the highest prediction accuracy and exhibits the best overall performance for this specific dataset

I.	INTRODUCTION

This statistic presents the assets of domestic banking groups in Portugal from 2008 to 2016. In 2016 the assets of Portuguese domestic banks amounted to 299 billion euros, which is a decrease of 14 billion euros from the previous year. The problem at hand involves predicting whether clients of a Portuguese banking institution will subscribe to a term deposit or not. The predictions are based on direct marketing campaigns conducted by the bank, which involve phone calls to the clients. The aim is to optimize the marketing strategy by identifying the clients most likely to subscribe to the term deposit, thereby increasing the success rate of the campaigns and improving the overall effectiveness of the bank's marketing efforts. In recent years, machine learning techniques have gained significant attention in the field of marketing analytics. Several studies have explored the application of classification algorithms to predict customer behavior and improve marketing campaign outcomes. There is a need to specifically investigate the performance of classification algorithms on the Bank Marketing dataset. The analysis will involve measuring classification accuracy, precision, recall, and F1-score to assess the predictive capabilities of each algorithm. Additionally, the study aims to identify the most effective algorithm for this specific dataset, providing actionable insights for financial institutions to optimize their marketing strategies.

II.	THE DATA SET

Dataset was obtained from the UCI Machine Learning repository. it contains 45211 instances each containing 17 attribute with no missing values. Understanding the relationships between the features in the bank marketing
 
dataset is crucial for developing effective classification models. For instance, the age of the client might play a role in their likelihood to subscribe to a term deposit. Similarly, the type of job or education level could influence their decision-making. Exploring the correlations and dependencies between these features can provide insights into the factors that impact the subscription behavior of the clients. The campaigns was based under the phone call where 16 attribute feature relating to the client details last 18th feature was represent by the client subscription Then we have 7 numeric datatype and 10 catagorical features. TABLE 1 display the summary of the data set describing features, feature types

TABLE 1. DATASET FEATURES

No	Description	Type
1	Age	Numeric
2	Job	Categorical
3	Martial status	Categorical
4	Education	Categorical
5	Credit default	Categorical
6	Housing loan	Categorical
7	Personal Loan	Categorical
8	Contact	Categorical
9	Month	Categorical
10	Contact day of week	Categorical
11	Duration	Numeric
12	Campaign	Numeric
13	Days passed	Numeric
14	Previous contact	Numeric
15	Outcome call	Categorical
16	Employment Variation rate	Numeric
17	Client Subscription	Numeric

III.DATA PREPARATION

This section covers data analysis on StandardScaler and dealing with categorical data and categorical data encoding, analysis of numerical values and datastandardisation




A.	StandardScaler

StandardScaler is a class in the scikit-learn toolkit that is used to standardise numerical properties in a dataset. It follows the previously specified data standardisation process. The attributes have a mean and standard deviation of 0 and 1, respectively, thanks to the StandardScaler's scaling. This transformation scales the features so that they can be used with machine learning techniques that are sensitive to feature scale.The mean and standard deviation of each feature are determined before StandardScaler is fitted to the 'bank' dataset using the fit() technique. The scaled_features array is then created by applying the standardisation to the dataset using the transform() method. producing the scaled_features array. After that, the array is changed back into a DataFrame


 

B.	Categorical Feature Encoding

Numerical data, as the name suggests, has features with only numbers (integers or floating-point). On the other hand, categorical data has variables that contain label values (text) and not numerical values.Machine learning algorithms typically require numeric input, so categorical variables need to be encoded as numerical values. This can be done through one-hot encoding, where each category is represented by a binary (0 or 1) variable. The pandas library provides the get_dummies() function to perform one-hot encoding. Categorical variables with multiple categories can be encoded as multiple binary variables, and one of the resulting variables can be dropped to avoid multicollinearity (the dummy variable trap).2.
 
TABLE 2 ENCODING CATEGORICAL FEATURES TO BINARY
            Fig 1. Standard Scalar
To overcome these three techniques were used as suggested	                   →	
by He. For source code see appendix IX.A.7)

1)	Random Under-sampli

Fig 1. Standard Scalar

The above correlation plot helps in understanding the relationship between each feature and the target variable. Positive values indicate a positive correlation, meaning that as the feature value increase. Negative values indicate a negative correlation, indicating an inverse relationship. By examining the correlation plot, you can identify which features have a strong influence on the target variable and prioritize them during feature selection or model training. The resulting correlation values are then sorted in descending order using the sort_values() method 

Then remaining data are get manipulated by loop over the columns to analysis entire dataset for visualization created as a plot 
    In our dataset we are using hot encoding function to convert categorical feature to the numerical for the following attributes for deposit, loan, marital, education, contact, month. For above martial status if we want to remove person who divorced 0 as dummy value by using the categorical encoding feature
	   TABLE 3 DUMMY VARIABLE REMOVAL
Marital Status	Numerical
Married	              1
  
C.	Data Standardisation
    Commonly the process of transforming numerical data to a        standard scale, typically with a mean of 0 and a standard deviation of 1. It is an essential step in data preprocessing that helps in removing the effects of different scales and units, making the data comparable and improving the performance of machine learning algorithms.Standardizing the data ensures that all features have the same scale and reduces the impact of outliers, which can skew the results. By bringing the features to a common scale, it allows the algorithm to give equal importance to each feature during the learning process.There are different methods available for data standardization, but one commonly used technique is called x-score normalization. The formula for z-score normalization is as follows

                           z = (x - mean) / standard deviation  
 In this paper, standardization was completed utilizing python Standard Scaler class from preprocessing module in scikit-learn library.
 
IV. MACHINE LEARNING CLASSIFICATION TECHNIQUES

A. K-Nearest Neighbor (K-NN)
K-Nearest Neighbor is a supervised machine learning algorithm that predicts the class or value of a new data point based on the classes or values of its K nearest neighbors in the training set. It operates on the similarity principle, which makes the assumption that data points with similar properties are probably members of the same class or have comparable values.The algorithm uses a distance metric, such as Euclidean distance, to determine the distance between the new data point and each of the other data points in the training set. The K closest neighbours with the shortest distances are then chosen.When performing classification tasks, the algorithm decides which class to assign to a new data point by polling its K nearest neighbours. The predicted label for the new data point is chosen to be the class label that appears the most frequently among the neighbours.For regression jobs, the method projects the new data point's value to be the average or median value among the target variable's K nearest neighbours.The following equation can be used to determine the Euclidean separation between two data points (x1, y1) and (x2, y2) in a two-dimensional feature space.

	Distance = √((x2 - x1)^2 + (y2 - y1)^2)

Overall, K-NN is a simple yet powerful algorithm that relies on the concept of similarity to make predictions. It is widely used in various fields for classification and     regression.
	 Fig.2   A. K-Nearest Neighbor (K-NN)
B. Logistic regression
A statistical classification procedure known as logistic regression is used to forecast the likelihood of a binary result based on one or more independent factors. It is frequently used in many different disciplines, including social sciences, machine learning, and statistics. In logistic regression, the independent variables may be continuous or categorical, and the dependent variable may be binary or categorical (e.g., Yes/No, True/False, 0/1, etc.). Finding the best-fit line or decision boundary between the two classes is the objective.The logistic function, also referred to as the sigmoid function, transfers every real-valued integer to a value between 0 and 1. The logistic regression model uses this function to determine the probability of the binary result. The logistic regression formula looks like this:

                            P(Y=1|X) = 1 / (1 + e^(-Z))

The chance that the dependent variable (Y) will be 1 given one or more independent variables (X) is known as P(Y=1|X). The independent variables' linear combinations, as weighted by their corresponding coefficients, are represented by Z.The natural logarithm's base is e as well. The simplicity, interpretability, and suitability for both continuous and categorical independent variables are only a few benefits of logistic regression. It does, however, presuppose a linear relationship between the independent factors and the dependent variable's log-odds to choose the best fit model. The Regression with polynomial features work better with normalization instead of standardization. 

		Fig.3 logistic regression  


C. Ensembles
	This learning technique combining multiple models. Thus a collection of models is used to make predictions rather than an individual model.

     1) Random Forest
Random Forest builds a "forest" of decision trees, where each tree is trained on a random subset of the data and features. This randomness is intended to increase variation among the trees and lessen overfitting. An ensemble learning system called Random Forest mixes various decision trees to produce predictions. Subsets of the training data and feature sets are chosen at random, and each tree is trained independently. In order to arrive at the final forecast, all of the predictions from the trees are combined, either through voting (for classification) or averaging (for regression). Based on a portion of the data and features, each decision tree in the forest generates its own predictions.

D. Support Vector Machine:

	SVM works by finding an optimal hyperplane that separates data points of different classes or predicts the target variable based on the given input features. It is particularly effective in cases where the data is not linearly separable.The hyperplane is defined by a subset of the training data points called support vectors. These support vectors are the critical data points that determine the position and orientation of the decision boundary.SVM works best with numerical data, so categorical variables are often encoded or transformed.generally recommended to scale the input features to ensure that they are on a similar scale. This helps SVM perform better and avoids bias towards features with larger magnitudes. SVM uses a kernel function to transform the input features into a higher-dimensional space, where the data may become more separable. Commonly used kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid.

Linear SVM:
	w^T * x + b = 0

Non-Linear SVM:
	f(x) = sign(sum(alpha_i * y_i * K(x_i, x)) + b)
 
Fig 4 SVM   Linear Separation

Fig 5 SVM Non-linear Separation Using Kernel

V. EXPERIMENT RESULTS
Using an HP Victus laptop with Ryzen 5 5000 series CPU and 16GB RAM, algorithms were implemented and results were obtained. The best-performing parameters, which were found using grid search, were used to train the models. For all models, confusion matrices, ROC curves (apart from SVM), and model metrics were obtained and are shown for each model Then each model was trained to evaluate to obtain the best result.
Model evaluation:
•	Model Accuracy
•	Model Precision
•	Support 
•	Recall
•	F1-score
•	Test and train score

A. K-Nearest Neighbor
Using python libraries to K-Nearest Neighbor classifier was trained and results presented below figure 5 , 6 and Fig.7 Hyperline Parameter

Table.4 Classification of Report:
Parameters	Precision	Recall	F1-Score	Support
Negative	0.91	0.98	0.94	11969
Positive	0.64	0.29	0.40	1595
Macro avg	0.77	0.63	0.67	13564
Weighted avg	0.88	0.90	0.88	13564
Table.5
Actually Accuracy	0.84
Test Accuracy	0.89
Train Accuracy	0.90
Test Recall	0.28
Train Recall	0.34
Where the best K value is 9 it means to calculate the nearest number among them. There is no misclassification around test case. Accuracy rate on KNN 0.84

B. Logistic regression.
	Using python libraries to Logistic regression classifier and with polynomial feature degree was trained and results presented in below table and figure
	 
		Fig 8.Confusion mtrix
Table.6 Classification of Report:
Parameters	Precision	Recall	F1-Score	Support
Negative	0.92	0.98	0.95	11969
Positive	0.68	0.35	0.47	1595
Macro avg	0.80	0.67	0.71	13564
Weighted avg	0.89	0.90	0.89	13564
Table.7
Actually Accuracy	0.91
Test Accuracy	0.90
Train Accuracy	0.90
Test Recall	0.35
Train Recall	0.33

(ii) Logistic Regression with the polynomial feature 
	Regression with polynomial feature works better with normalization instead of standardization so working with the degree of 2.


     Fig.9 Confusion matrix
	Table.8 Classification of Report:
Parameters	Precision	Recall	F1-Score	Support
Negative	0.92	0.97	0.95	11969
Positive	0.66	0.41	0.50	1595
Macro avg	0.79	0.69	0.73	13564
Weighted avg	0.89	0.91	0.90	13564
	Table.9
Actually Accuracy	0.91
Test Accuracy	0.90
Train Accuracy	0.90
Test Recall	0.35
Train Recall	0.33
	
As we expected logistic regression with polynomial feature works is more effective than the normal logistic regression and delivered the best result. When comparing to KNN algorithm logistic regression gives more accuracy.

C. Support Vector Machine 
	Using python libraries SVC classifier, support vector machine model was trained and results presented below 
 
Fig.10 Confusion matrix
Table.10 Classification of Report:
Parameters	Precision	Recall	F1-Score	Support
Negative	0.92	0.97	0.95	11969
Positive	0.66	0.41	0.50	1595
Macro avg	0.79	0.69	0.73	13564
Weighted avg	0.89	0.91	0.90	13564
Table.11 
Actually Accuracy	0.92
Test Accuracy	0.90
Train Accuracy	0.91
Test Recall	0.42
Train Recall	0.44

The accuracy result is much more better than KNN but not so far as logistic regression .The dataset is complex, according to the results, and simple models like support vector machines have a hard time learning and making accurate predictions. for running source code refer to appendices for the modeling function and production result.

D. Random Forest
	Random forest algorithm from the scikit-learn repository in python was trained, tested and results obtained. Confusion matrices presented in Fig. 11 and results summary presented in TABLE 12

 Fig11.Confusion matrix
 Random Forest delivered the excellent accuracy results and significantly outperform those from KNN,SVM and Logistic Regression 

VI. CONCLUSION

Random forest performed best across all the model.The overall performance level of all techniques plotted as one graph by ROC curve analysis

decision tree models, combining their classification results and using majority voting to arrive at final prediction (Gra̧bczewski 2014).
1)	Adaptive Boosting (ADA Boost)

Review of Random Forest ensemble model result produced the best result in most areas especially in the time taken to process. Random Forest were more than three time faster when compared to other classification techniques used .
	 Based on result obtained the best model to use for the prediction of client subscription is Random Forest with dataset that taken from the UCI. 
 
Actually Accuracy	0.93
Test Accuracy	0.90
Train Accuracy	0.99
Test Recall	0.43
Train Recall	1.0
Parameters	Precision	Recall	F1-Score	Support
Negative	0.92	0.97	0.95	11969
Positive	0.67	0.40	0.50	1595
Macro avg	0.80	0.69	0.73	13564
Weighted avg	0.89	0.91	0.90	13564

