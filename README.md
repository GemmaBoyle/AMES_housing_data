# Predicting house prices with the AMES housing data set

## Table of Contents

1. [Objectives](##Objectives)
2. [Background](##Background)
3. [Data Cleaning and Feature Engineering](##Data_Cleaning_and_Feature_Engineering)
4. [Exploratory Data Analysis](##Exploratory_Data_Analysis)
5. [Modelling](##Modelling)
6. [Evaluation](##Evaluation)
7. [Limitations](##Limitations)
8. [Conclusions and decision recommendations](##Conclusions_and_decision_recommendations)
9. [Further investigations](##Further_investigations)
10. [Key learnings](##Key_learnings)

## Objectives
1. Develop an algorithm to reliably estimate the value of residential houses based on *fixed* characteristics.
2. Identify characteristics of houses that the company can cost-effectively change/renovate with their construction team.
3. Evaluate the mean dollar value of different renovations.
4. Determine which features predict the abnormal category in the Sale Condition feature (classification with massive class imbalance).

## Background

Situation: A real estate company interested in using data science to determine the best properties to buy and resell. Specifically, they would like to identify the characteristics of residential houses that estimate the sale price and the cost-effectiveness of doing renovations.

There are three components to the project:

1. Estimate the sale price of properties based on their "fixed" characteristics, such as neighbourhood, lot size, number of stories, etc.
2. Estimate the value of possible changes and renovations to properties from the variation in sale price not explained by the fixed characteristics. Your goal is to estimate the potential return on investment (and how much you should be willing to pay contractors) when making specific improvements to properties.
3. Determine the features in the housing data that best predict "abnormal" sales (foreclosures, etc.)

## Data Cleaning and Feature Engineering

The data required a number of pre-processing steps before the execution of machine learning techniques
1. Remove non-residential properties
2. Remove rows with Null values for relevant features 
3. Convert numerical categorical data to string types for dummification
4. Manually split the features into fixed / non-fixed

## Exploratory Data Analysis

![Histogram](https://github.com/GemmaBoyle/AMES_housing_data/blob/main/Images/Distribution_of_house_prices.png)

The distribution of Sale Prices show a distinct positive skew, with the majority of the data concentrated around the 150,000 mark and less of the more expensive properties. 

Equally, the box plot shows a number of significant outliers on the upper end, being more than 1.5 times larger than the interquartile range of the data. I did consider whether to remove a number of the more extreme outliers, but after running models this did not improve scores and hence all data points were eventually included. 

## Modelling 

#### Fixed features
Several machine learning algorithms were first applied to the fixed (non-renovatable) house characteristics (Basic Linear Regression / Lasso regularization / Ridge regularlization / Elastic Net regularization) with the Elastic Net model running with the highest mean-CV R2 score of 77% (Training Score: 83%, Test Score: 85%) at a cross-validated alpha-ratio of 0.999. 

#### Non fixed features

The same models were run on the residuals of the Sale Prices from the fixed features analysis above to see how much, if any, of the variance from the first model could be explained by the non-fixed features of the properties. Although the Lasso model came out with the largest mean CV score it did not generalise well as it produced a negative R2 score. The Elastic Net ratio came out as 1, meaning a full Lasso model. The Ridge model had a slightly lower CV score than the Lasso but did generalise better, hence this is my preferred model.

#### Abnormal sales

This part of the project suffered from massive class imbalance (the majority class accounted for 93% of the data) and undersampling was used to address this and give a baseline of around 53%. Logistic regression and a K-neighbors algorithm were utilised on this data. The mean CV score based on a KNN gridsearch using the features identified by the Lasso model above gave the best mean CV score of 0.72, well above the baseline of 0.56. However, these results must be taken with caution, as although there were positive CV and training scores on the training data, the test score was marginally above the baseline, suggesting the model will not generalise well.  

In this case I would suggest the Ridge penalty logistic regression model above is preferable as although the mean CV score was lower at 0.658, the training and test scores were more sensible, indicating that this model generalises better and would be a better future indicator of abnormal sales. However, the feature outcomes of this model make little sense, with a sale type COD (court officer deed) counting against the abnormal sales, and having an attached garage counting towards abnormal sales. 


## Evaluation

#### Fixed Features

* Feature importance graph

The largest contributor to Sale Price according to the best model (and several other models) was the area of above ground living area, adding close to £11,00 value to the house sale price. The 10 most important features were all positive contributors to the outcome, many to do with the square footage of the property (ground floor footage, first floor footage, second floor footage) or the neighbourhood of the property, as shown. The importance of the features is given in dollar amounts along the horizontal axis. 

![Predicted V Real](https://github.com/GemmaBoyle/AMES_housing_data/blob/main/Images/Predictions%20against%20true%20sale%20prices.png)

A plot of the predicted price against real price shows a good positive linear correlation, with some irregularities on the upper end of the data, likely stemming from there not being enough data available for higher-priced properties. 

#### Non-fixed features

* Non-fixed feature importance 

The largest contributor to the variance in residuals is clay tiles, this is a negative relationship meaning that removing clay tiles from a house can add almost 40000 to the sale price of the house, other features being fixed. The largest positive contributor to the variance in residuals is full bath 3, meaning that having three above grade bathrooms contributes around 2200 to the house sale price. 

#### Abnormal sales

I would suggest that predicting abnormal sales does not rely on the features of the house, instead it is perhaps more reasonable to assume that these sales are based on the features of the seller and their personal financial situations rather than house properties. More data gathering is needed to investigate this hypothesis. 

## Limitations

The fixed-features algorithms consistently showed lower mean CV scores, suggesting the model does not test well on unseen data. This points to overfitting on the model, meaning that there may be collinearity in the data and some feature pruning is required. 

The important features identified when trying to predict abnormal sales did not make sense in the context of the problem, more work is needed to understand why this is the case. 

## Conclusions and decision recommendations

The first model shows the main features the company should look for in a house for it to have a higher sale value, a large above ground living area, a basement of large height, a large first floor area, a large garage area and a large basement area. As we can see these features are based on the size of the house, hence when buying property, bigger is better! It is important to note that the R2 squared value for the best model is around 0.75, meaning the model only accounts for around 75% of the variance between actual and predicted prices. 

The second model was run on the residuals of the best model from the fixed features, we can see that even in the best case scenario only around 17% of the difference between actual and predicted price can be explained by recourse to the non-fixed features of the property, the rest being accounted for by the imperfectness of our model from part 1.

Hence I would not recommend buying houses in order to 'fix them up' and expect to do particularly well. However, where houses have already been purchased I would recommend removing clay tiles from the roof and replacing with wooden shingles, increasing the quality of bathrooms and increasing the overall quality and condition of the property, as cheaply as possible, to add some value to the Sale Price

Construction and labour would be better served by building an extension / loft conversion in the house, where available and where planning permission would be granted, in order to increase the overall living area of the property, as this is the largest indicator of sale price.

I would not be comfortable using the second model to choose which houses to buy and renovate, instead I would focus on buying larger properties.

## Further investigations

1. The plot of actual prices against predicted prices for the non-fixed features set showed a quadratic shape and hence may benefit from polynomial features modelling. 
2. There was more opportunity for feature engineering here than time allowed, for example would it have been productive to combine all square footage measures into a single figure to use as a predictor? 

## Key learnings

1. The project would have benefited considerably from more EDA, particularly running a correlation heat-map for continuous features against the Sale Price target. This would have allowed for a decrease in the number of features used, in order to prevent overfitting, and allowed us to check for collinearity in the data in order to appropriately choose features. 
2. The problem of class imbalance was interesting and led to a lot of research on my part about where these issues arise and possible solutions. This is a continuing area of interest. 



