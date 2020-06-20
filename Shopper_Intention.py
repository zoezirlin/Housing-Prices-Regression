#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:55:53 2020

@author: zoezirlin
"""
#------------------------------------------------------------------------------
# Importing relevant libraries

import pandas as pd #Data manipulation and analysis
import numpy as np #Data manipulation and analysis
import seaborn as sns #Data visualization
from matplotlib import pyplot as plt #Data visualization
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix #Testing the logistic regression
from scipy.stats import pointbiserialr #importing point biserial correlation abilities
#conda install -c conda-forge pandas-profiling
#from pandas_profiling import ProfileReport




#------------------------------------------------------------------------------
### Importing and learning the basics of the dataset ###



## Importing the dataset from desktop
intention_df = pd.read_csv("/Users/zoezirlin/Desktop/online_shoppers_intention.csv")



## Printing the first ten datalines in dataframe "intention"
intention_df[:10]





#------------------------------------------------------------------------------
### Graphing and Data Analysis ###



## Learning that 84% of instances/observations did NOT result in purchase
intention_df["Revenue"].value_counts(normalize=True).plot(kind= 'bar', color = 'Orange')
plt.title('Revenue Disparity')
intention_df["Revenue"].value_counts(normalize=True)



## Learning that the bounce rates do not follow a normal distribution, grouped at rates of 0 to 0.05
intention_df["BounceRates"].plot.hist(grid=True, bins=10, rwidth=2, color= 'Orange')
plt.xlabel('Bouce Rate')
plt.title('Bouce Rates Histogram')



## Learning that the exit rates do not follow a normal distribution...
# but moreso than bouce rates, grouped at rates of 0 to 0.075
intention_df["ExitRates"].plot.hist(grid=True, bins=10, rwidth=2, color= 'Orange')
plt.xlabel('Exit Rate')
plt.title('Exit Rates Histogram')



## Creating bar plots for categorical variables
cols = ['TrafficType','Region','VisitorType','Month']

n_rows = 2
n_cols = 2

figs, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 5, n_rows * 5)) #the 5 denotes size of the graph

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r * n_cols + c
        ax = axs[r][c]
        sns.countplot(intention_df[cols[i]], hue=intention_df['Revenue'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Revenue', loc = 'upper right')

plt.tight_layout()



## Looking at revenue by month
sns.barplot(x = 'Revenue', y = 'Month', data = intention_df )
intention_df.pivot_table('Revenue','Month')
#November and October have the highest revenue probabilities
#February has the lowest revenue probabilities


## Looking at revenue by month and weekend
month_pivot_plot = intention_df.pivot_table('Revenue', 
                        index = 'Weekend',
                        columns = 'Month').plot()
month_pivot = intention_df.pivot_table('Revenue', 
                        index = 'Weekend',
                        columns = 'Month')
# Weekends in November have the highest revenue ratios
# Weekends in February have the lowest revenue ratios
# Weekdays in November have the highest revenue ratios
# Weekdays in February have the lowest revenue ratios



## Looking at purchase rates based on visitor type
# Learning that New Visitors have a higher rate of purchase than Return Visitors
intention_df["VisitorType"].value_counts().plot(kind='bar', color='Pink')
intention_df.pivot_table('Revenue','VisitorType')



## Looking at chances of purchase by region and weekday/weekend             
intention_df.pivot_table('Revenue', 
                     index = 'Weekend' , 
                     columns = 'Region')
# Learning that on a weekend in region 2, there is a 22% chance of purchase
# Learning that on a weekend in region 9, there is a 20% chance of purchase
# Learning that on a weekday in region 8, there is a 12% chance of purchase    



## Looking at bouce rates by region
ax = sns.boxplot(x='Region', y='BounceRates', data=intention_df)
x = intention_df['Region']
y = intention_df['BounceRates']



## Looking at product related duratin by revenue
ax = sns.boxplot(x='Revenue', y='ProductRelated_Duration', data=intention_df)
x = intention_df['Revenue']
y = intention_df['ProductRelated_Duration']





#------------------------------------------------------------------------------
### Variable Logistic Regression Elimination/Model Creation ###



## Correlation Matrix
corr_matrix = intention_df.corr(method='pearson')



## Correlation Matrix Graphic
corr = intention_df.corr()
heatmap = sns.heatmap(
    corr, 
    square=True
)





#------------------------------------------------------------------------------
### Logistic Regression ###



 # Model: 'Revenue' = 'ExitRates' : 8
                   # 'BounceRates' : 7
                   # 'PageValues' : 9
                   # 'ProductRelated' : 5
                   # 'ProductRelated_Duration : 6
                   # 'Administrative : 1



## Dropping NA observations
intention_df = intention_df.dropna()
intention_df.head()



## Assigning the True/False options of Revenue to 0/1
intention_df['Revenue'] = pd.get_dummies(intention_df['Revenue'])

intention_df['Weekend'] = pd.get_dummies(intention_df['Weekend'])



## Assigning the New/Returning options of VisitorType to 0/1
intention_df['VisitorType'] = pd.get_dummies(intention_df['VisitorType'])



## Splitting dataset into response and predictor variables
#predictor_cols = ['ExitRates',
  #                'PageValues',
   #               'ProductRelated',
    #              'ProductRelated_Duration',
     #             ]

#X = intention_df[predictor_cols]
#y = intention_df.Revenue









## Logistic Regression        

X = intention_df['ProductRelated_Duration']
y = intention_df.Weekend
X = sm.add_constant(X)

model_1 = sm.Logit(y,X, method='nm')
result = model_1.fit()
result.summary()

# use chi squared test instead...?




#profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
#profile = ProfileReport(intention_df, title = 'Profiling Report', explorative=True)

















