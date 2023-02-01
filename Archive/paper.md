---
title: Discover Malicious Websites Using Data Mining Algorithms
date: November 22, 2021
author: Sai Gowtham, Srinivas Gutta, Manish Mapakshi, Pratibha Awasthi, San JosÃ©️ State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract


# Introduction

Phishing is a deceptive practice in which an attacker attempts to obtain sensitive information from a victim. Emails, text messages, and websites are commonly used in these types of assaults. Phishing websites, which are on the rise these days, have the same appearance as real websites. Their backend, on the other hand, is geared to harvest sensitive information provided by the victim. The machine learning community, which has constructed models and performed classifications of phishing websites, has recently become interested in discovering and detecting phishing websites. This research includes two dataset versions with 58,645 and 88,647 websites categorized as real or fraudulent, respectively. These files contain a collection of legitimate and phishing website examples. Each website is identified by a collection of characteristics that indicate whether it is real or not. Thus, data can be used as a source of information in the machine learning process.

# Data Description

The data in this presentation was gathered and compiled to develop and analyze several categorization algorithms for detecting phishing websites using URL characteristics, URL resolving metrics and external services. Six groups of attributes can be found in the prepared dataset:  

* attributes based on the whole URL properties 
* attributes based on the domain properties 
* attributes based on the URL directory properties 
* attributes based on the URL file properties
* attributes based on the URL parameter properties
* attributes based on the URL resolving data and external metrics.

As shown in Figure 1, the first group is based on the values of the characteristics on the entire URL string, but the values of the next four groups are based on specific sub-strings. The final set of attributes is based on URL resolve metrics as well as external services like Google's search index.

![Separation of the whole URL string into sub-strings](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/SeparationofthewholeURLstringintosubstrings.jpg?raw=true)

The dataset has 111 features in total, except the target phishing attribute, which indicates if the instance is legitimate (value 0) or phishing (value 1).  We have two versions of the dataset, one with 58,645 occurrences with a more or less balanced balance between the target groups, with 30,647 instances categorized as phishing websites and 27,998 instances labeled as legitimate. The second dataset has 88,647 cases, with 30,647 instances tagged as phishing and 58,000 instances identified as valid, with the goal of simulating a real-world situation in which there are more legitimate websites present. We have used dataset_small for further analysis and model building as it has more balanced classes.

<img src="https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/Thedistributionbetweenclassesforbothdatasetvariations.jpg?raw=true" width="500" height="500" >


# Methods

## Method 1:

## Data Preprocessing:

It is the process of transforming raw data into an understandable format. Data pre-processing is used to enhance the quality of the data for future modeling purposes. Our dataset consists of imbalanced data. Several methods have been used in this step to clean and increase the quality of the data. The methods are as follows:

### Feature Selection using Variance Threshold:
There are several features that consist of duplicate values in it’s columns. These values have to be dropped to reduce the dimensionality. We have used a method called “Variance Threshold” for feature selection. Variance Threshold sets up a threshold value and any feature whose variance doesn’t meet the threshold. By default, it removes all the zero-variance features. In this model, there are 13 features which don't meet the threshold and these features are dropped.

### Eliminating Missing Values:
Most of the features in the data have “-1” as a value. URL attributes can never have negative values. For example, features like quantity, length and params can never be negative. It is highly illogical to consider these negative values. These “-1” values are considered as missing values here. We calculated the percentage of “-1”  values in each and every feature. Features which have more than 80% of its values as “-1” are dropped. Later, all the other features consisting of “-1” values are replaced with “NAN”. As you can see in the figure below, the missing number library is used to plot all the missing values. It can clearly be noted that a lot of params values are missing. All the “params” features consisting of missing values are dropped.

<img src="https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/missingdata.png?raw=true" width="500" height="500" >

From the above figure, we can understand that some missing values still exist. To deal with these missing data we can use different imputation techniques such as Mean/Median/Mode imputation or use imputer algorithms like KNNimputer(), MissForest().  We have decided to approach this problem by using both the Mean impuation and KNN imputation and later compare results.

## KNN Imputed Data Analysis:
1. The Null data is imputed using KNN imputer with n_neighbhors:3. The distance measure used is euclidean distance.
2. Stratified split of this data to train and test data with test data size as 25%.
3. Standardizing the data using Standardscaler().
4. Three classifiers namely Logistic Regression, RandomForestClassifier, and XGBoost classifier are implemented to analyze this    data.
5. We are initializing these 3 classifiers with default parameters. Hyperparameter tuning will be done in the next phase after    feature selection.

Below figures are the metrics obtained after training the models and scoring on test data.

* Logistic Regression performance metrics:

![Logistic Regression performance metrics](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/metrics_lr.png?raw=true)

* XGBoost performance metrics: 

![XGBoost performance metrics](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/xgboostmetrics.png)

* RandomForest performance metrics:

![RandomForest performance metrics](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/randomforestmetrics.png)

### Feature Importance & Selection:
* Examining the model's coefficients is the simplest technique to analyze feature importances. It has some influence on the forecast if the assigned coefficient is a large (negative or positive) number. If the coefficient is zero, on the other hand, it has no bearing on the forecast.For Logistic Regression the feature importances is derived from their respective coefficients.

* RandomForest Classifier & XGBoost Classifier have built-in feature importance. The decrease in node impurity is weighted by the likelihood of accessing that node to compute feature significance. The number of samples that reach the node divided by the total number of samples yields the node probability. The more significant the feature, the higher the value.

Below are figures of feature importances using 3 models:

* Feature Importance Logistic Regression

![feature importance logistic_regression](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/featureimportancelogisticregression.png)

* Feature Importance XGBOOST

![feature importance XGboost](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/featureimportanceXGboost.png)

* Feature Importance RandomForest

![feature importance randomforest](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/featureimportancerandomforest.png)

* Implementing Recursive Feature Elimination (RFECV) to obtain optimal features. This algorithm calculates the importance of each feature in a recursive manner, then discards the least important feature. It begins by determining the relevance of each column's feature. It then deletes the column with the lowest relevance score and repeats the process. 

Parameters:
`estimator_: RandomForestClassifier(); cv_: StratifiedKFold(3); Scoring_:’accuracy’`

![Cross-validation vs features selected RFECV](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/RFECV.png)

From the above figure, We can observe that the cross-validation accuracy score doesn't change after selecting 30 features; it flattens as the number of features selected increases. Training our models on these top 30 features should increase the performance of the model.

## Mean Imputed Data Analysis:
In this Mean Imputed Data Analysis part we have calculated the Mean of each column and have replaced the NAN values with those values.In future we plan to impute the data using different techniques.So, once the data is as been imputed with the Mean we try to train the XGBoost Classifier using the imputed data inorder to find the important features.

* Feature Importance XGBOOST

 ![feature importance XGboost](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/XGBoost_featureImportance.png)

From the above graph we can clearly depict the important features and then we consider the first 9 features because after that we can clearly see that the importance becomes constant.

### Analysis of the Important Features:
Now we do the basic analysis on the features which we have selected and then try to find the distribution of data and then we try to Feature engineering.So, at first we plot the box plots inorder to depict the Outliers.

![BoxPlot](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/Boxplot_for_outliers.png)

we can clearly see that the outliers exist in the above features like length_url,qty_slash_url. Now we try to plot the Violin plot for the remaining features inorder to find the distribution of the data.

![Violinplot](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/violin_plot.png)

#### Outliers: 
Earlier we observed the outliers on the selected features.So, now we try to remove the outliers from the selected columns using the Inter Quartile Range (IQR) methodology.

#### Correlation:
After the data has been modified we try to find the correlation between the features and the target variable 'Phishing'. We can use heatMap to visualise this.

![HeatMap](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/Heatmap.PNG)

From the above graph we can clearly say that there a negative correlation between for the feature time_domain_activation and phishing, a positive correlation between features like qty_slash_url and phishing is found.

#### Relativity:
Here we try to find the relation between the length features and the phishing feature.
![](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/directory_length_trend.png)

![](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/length_url_trend.png)

From the above graph can clearly see that as the as the length of URL, Directory Length increases it is most likely to be a phishing website.

By the Below scatter plot we can see that the probability of URl is phishing if the value lies between 0 and 5 for the features qty_percent_file,qty_hyphen_directory

![](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/scatter_plot%201.png?raw=true)

Likewise the below scatter plot we can see that the probability of URl is phishing is more as the value of qty_slash_url is more than 4.

![](https://github.com/SaiGowtham-11/Discover-Malicious-Websites-Using-Data-Mining-Algorithms/blob/main/images/scatter_plot%202.png)

Modeling: 
The accuracy is retrieved using different models. The first two models were trained on Mean imputed data while the other models were trained on KNN imputed data. They are:  
* Logistic Regression:
It is used in statistical software to understand the relationship between the dependent variable and one or more independent variables by estimating probabilities using a logistic regression equation.
The accuracy found through logistic regression is 85.6% 

* KNeighbors Classifier:
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. 
The accuracy found is 89.61%

* Decision Tree:
Decision tree is a predictive modelling approaches used in data mining.  It is constructed through algorithmic approach that identifies differents methods of splitting a data set based on different conditions. 
The accuracy found through decision tree is 95.65%

* Random Forest:
When a large number of decision tree operate as an ensemble, they make up Random Forest. Each tree in the random forest produces a class prediction, and the class with the most votes becomes the prediction of our model. The accuracy found through Random Forest is 97.52% 

* KNN:
KNN is a machine learning algorithm based on Supervised Learning technique. It assumes similarity between new data and available data and put new data into category that seems most similar to available categories. 
The accuracy found is 97.29%



It is concluded that the KNeighbors for KNN imputed data detected the best accuracy. 




# Comparisons

# Example Analysis

# Conclusions


# References



