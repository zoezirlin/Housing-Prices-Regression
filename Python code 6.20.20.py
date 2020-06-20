
"""
Created on Sat Jun  6 19:27:32 2020

@author: zoezirlin
"""
#------------------------------------------------------------------------------
### Importing my libraries

import numpy as np #important numpy
import pandas as pd #importing pandas
import seaborn as sns #importing seaborn
import scipy.stats as stats
from scipy.stats import pearsonr #importing pearson correlation abilities
from matplotlib import pyplot as plt #importing pyplot
import statsmodels.api as sm #importing statsmodels
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import power_transform
from statsmodels.formula.api import ols




#-----------------------------------------------------------------------------
### Importing and Learning Basics of the Data ###



## Importing train csv
train = pd.read_csv("/Users/zoezirlin/Desktop/house-prices-advanced-regression-techniques/train.csv")
## Importing test csv
test = pd.read_csv("/Users/zoezirlin/Desktop/house-prices-advanced-regression-techniques/test.csv")



## Printing the data set's first twelve observations
train[:12]



## Printing the shape of the train data
train.shape
# Learning that there are 1460 obervations and 81 attributes, 80 predictors...
#1 response (house price) in the training set



## Printing the shape of the test data
test.shape
## Learning that there are 1459 observations and 80 attributes in the test set... 
# (No response variable/ house price)





#------------------------------------------------------------------------------
### Feature Engineering ###


## Learning the non-null counts and datatypes for the train dataset
train.info()
# 72, 73, 74, 57, 6 have tons of missing data, will delete from dataset



## Removing largely absent columns
# Now 74 predictor variables/columns
train = train.drop(columns = ['Alley', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu'])
test = test.drop(columns = ['Alley', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu',])



## Removing observations with NaN 
train = train.dropna()





#------------------------------------------------------------------------------
### Continuous Variable Preliminary Analysis ###



## Attaining descriptive statistics for numerical predictor variables
train_description = train.describe()
# Average house price is 180_921.195890 with a SD of 79_442.502883
# Minimum house price is 34_900.000000 and maximum house price is 755_000.000000



## Creating a scatterplot for Overall Quality by Sales Price
ax1 = sns.scatterplot(x="OverallQual", 
                      y="SalePrice", data=train, color='mediumpurple')
plt.title("Overall Quality by Sales Price")



## Creating a scatterplot for Total Sq Ft of Basement Area by Sales Price
ax2 = sns.scatterplot(x="TotalBsmtSF", 
                      y="SalePrice", data=train, color='darkorchid')
plt.title("Total Square Feet of Basement Area by Sales Price")



## Creating a scatterplot for First Floor Sq Ft by Sales Price
ax3 = sns.scatterplot(x="1stFlrSF", 
                      y="SalePrice", data=train, color='plum')
plt.title("First Floor Square Feet by Sales Price")



## Creating a scatterplot for Above Ground Living Area by Sales Price
ax4 = sns.scatterplot(x="GrLivArea", 
                      y="SalePrice", data=train, color='m')
plt.title("Above Grade (ground) Living Area Square Feet by Sales Price")



## Creating a scatterplot for Garage Car Capacity Area by Sales Price
ax5 = sns.scatterplot(x="GarageCars", 
                      y="SalePrice", data=train, color='mediumvioletred')
plt.title("Size of Garage in Car Capacity by Sales Price")



## Creating a scatterplot for Garage Size Sq Ft by Sales Price
ax6 = sns.scatterplot(x="GarageArea", 
                      y="SalePrice", data=train, color='palevioletred')
plt.title("Size of Garage in Square Feet by Sales Price")



## Creating a correlation matrix
# Looking at variables that are strongly correlated with housing price
# Will subsequently check for multicollinearity
cont_corr_table = train.corr(method='pearson')



## Below are the continuous variables that have a linear relationship > .6
# OverallQual - .790982 (Overall material and finish quality)
# TotalBsmtSF - 0.613581 (Total square feet of basement area)
# 1stFlrSF - 0.605852 (First Floor square feet) (
#       GET RID OF THIS BC MULTICOLLINEARITY W TotalBsmtSF)
# GrLivArea - 0.708624 (Above grade (ground) living area square feet)
# GarageCars - 0.640409 (Size of garage in car capacity)





#------------------------------------------------------------------------------
### Continuous Variable Regression Analysis ###



## MODEL_1 (Multivariate Regression)
# With a constant

X = train[["OverallQual", "TotalBsmtSF", "GrLivArea", "GarageCars"]]
y = train["SalePrice"]
X = sm.add_constant(X) ## adds an intercept (beta_0) to our model

# Make the predictions by the model
model_1 = sm.OLS(y, X).fit()
predictions_1 = model_1.predict(X)

## Printing out the statistics
model_1.summary()

# Results:
# - Adj. R2 = 0.76
# - All variables are significant



# Figuring out multicollinearity problem
# 1. OverallQual x TotalBsmtSF = 0.5378084986123927
# 2. OverallQual x 1stFlrSF = 0.4762238290781775
# 3. OverallQual x GrLivArea = 0.5930074300286511
# 4. OverallQual x GarageCars =	0.6006707165907189
# * 5. TotalBsmtSF x 1stFlrSF = 0.8195299750050355
# 6. TotalBsmtSF x GrLivArea = 0.4548682025479028
# 7. TotalBsmtSF x GarageCars = 0.43458483429168826
# 8. 1stFlrSF x GrLivArea = 0.5660239689357487
# 9. 1stFlrSF x GarageCars = 0.43931680799067063
# 10. GrLivArea x GarageCars = 0.46724741879518455



## MODEL_2 (Multivariate Regression)
# With a constant

X = train[["OverallQual", "TotalBsmtSF", "GrLivArea"]]
y = train["SalePrice"]
X = sm.add_constant(X) ## adds an intercept (beta_0) to our model

model_2 = sm.OLS(y, X).fit()
predictions_2 = model_2.predict(X) # make the predictions by the model

# Print out the statistics
model_2.summary()
# Results:
# - Adj. R2 =  0.742
# - All variables are significant
# - This is the better model bc less predictors for similar adj. R2





#------------------------------------------------------------------------------
### Categorical Variable Preliminary Analysis ###


## Categorical Variables of Interest (Qualtitatively found): 
# 1. PavedDrive
# 2. GarageFinish
# 3. Functional
# 4. KitchenQual
# 5. HeatingQC (may have multicollinearity with below CentralAir)
# 6. CentralAir
# 7. ExterQual (may have multicollinearity with below ExterCond)
# 8. ExterCond
# 9. BldgType



## Creating new dataset with just the prospective categorical variables
# Dummy variables
train_cat = train.loc[: ,['PavedDrive','GarageFinish','Functional',
                         'KitchenQual','HeatingQC','CentralAir',
                          'ExterQual','ExterCond','BldgType',
                          'SalePrice']]

train_cat['PavedDrive'] = train_cat.PavedDrive.map({'Y':1,
                                                'N':0
                                                })

train_cat['GarageFinish'] = train_cat.GarageFinish.map({'Fin':4,
                                                    'RFn':3,
                                                    'Unf':2,
                                                    'NA':1
                                                    })

train_cat['Functional'] = train_cat.Functional.map({'Sal':1,
                                                'Sev':2,
                                                'Maj2':3,
                                                'Maj1':4,
                                                'Mod':5,
                                                'Min2':6,
                                                'Min1':7,
                                                'Typ':8
                                                })

train_cat['KitchenQual'] = train_cat.KitchenQual.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })

train_cat['HeatingQC'] = train_cat.HeatingQC.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })

train_cat['CentralAir'] = train_cat.CentralAir.map({'Y':1,
                                                'N':0
                                                })

train_cat['ExterQual'] = train_cat.ExterQual.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })

train_cat['ExterCond'] = train_cat.ExterCond.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })

train_cat['BldgType'] = train_cat.BldgType.map({'TwnhsI':1,
                                                'TwnhsE':2,
                                                'Duplx':3,
                                                '2FmCon':4,
                                                '1Fam':5
                                                })



## Dropping the NaN values
train_cat = train_cat.dropna()



## Running correlations between predictors and response
cat_corr_table = train_cat.corr(method = 'pearson')

## Below are the categorical variables that have a linear relationship > .6
# ExterQual - 0.6955084018095976 ()
# KitchenQual - 0.6712135411598092 ()



## Boxplot for Exterior Quality by Sales Price
ax_b = sns.boxplot(x="ExterQual", y="SalePrice", data=train)
plt.title("Exterior Quality by Sales Price")



## Boxplot for Basement Expoosure by Sales Price
ax_c = sns.boxplot(x="BsmtExposure", y="SalePrice", data=train)
plt.title("Exterior Quality by Sales Price")


          
## Pivot Table for Basement Exposure
# 59% of obs no basement exposure
sm_pivot = train.pivot_table('SalePrice', columns= 'BsmtExposure', 
                  aggfunc = lambda x:x.sum()/train['SalePrice'].sum())



## Pivot Table for 
# 83% of obs RL MSzoning
mszon_pivot = train.pivot_table('SalePrice', columns= 'MSZoning', 
                  aggfunc = lambda x:x.sum()/train['SalePrice'].sum())



## Pivot Table 
# 57% of obs reg lot shape, and 37% IR1
lotshape_pivot = train.pivot_table('SalePrice', columns= 'LotShape', 
                 aggfunc = lambda x:x.sum()/train['SalePrice'].sum())






# consider running some ANOVA's on these


# sales price by the categorical variables of interest
# 'SalePrice' by 'LotShape'

#lm_1 = ols('SalePrice ~ LotShape', data = train_cat).fit()
#table_lm_1 = sm.stats.anova_lm(lm_1)
#print(table_lm_1)







#------------------------------------------------------------------------------
### Final Regression Modeling ###



## Composing final dataset with significant variables
train_final = train.loc[: ,["OverallQual", 
                            "TotalBsmtSF", 
                            "GrLivArea",
                            "ExterQual",
                            "KitchenQual",
                            "SalePrice"
                            ]]



## Recoding the two significant categorical variables
train_final['ExterQual'] = train_final.ExterQual.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })

train_final['KitchenQual'] = train_final.KitchenQual.map({'Po':1,
                                                  'Fa':2,
                                                  'TA':3,
                                                  'Gd':4,
                                                  'Ex':5
                                                  })



## MODEL_3 (Multivariate Regression)
# With a constant

X = train_final[["OverallQual", "TotalBsmtSF", "GrLivArea", "ExterQual", "KitchenQual"]]
y = train_final["SalePrice"]
X = sm.add_constant(X)

model_3 = sm.OLS(y,X).fit()
predictions_3 = model_3.predict(X)

model_3.summary()
# - Adj. R2 of 0.758



## Checking for multicollinearity
final_corr_table = train_final.corr(method = 'pearson')
# Overll quality and exterior quality are somewhat correlated (abt .7)
# Deciding to remove ext. quality variable to ammend multicollinearity



## MODEL_4 (Multivariate Regression) 
# With a constant

X = train_final[["OverallQual", "TotalBsmtSF", "GrLivArea", "KitchenQual"]]
y = train_final["SalePrice"]
X = sm.add_constant(X)

model_4 = sm.OLS(y,X).fit()
predictions_4 = model_4.predict(X)

model_4.summary()
# Results: Adj. R2 of 0.755 with constant, strongest regression model



## Partial Regression Plots
# Total square feet of basement area not great...
# but model still better when it is included
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model_4, fig=fig)



## Residual Plots for Each Variable
# Showing megaphone effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final.TotalBsmtSF, 
              train_final.SalePrice).set_title('BSMT W influential')

# Showing megaphone effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final.GrLivArea, 
              train_final.SalePrice).set_title('GRL W influential')

# Not really nonconstant variance :/
plt.subplots(figsize = (12,8))
sns.residplot(train_final.KitchenQual, 
              train_final.SalePrice).set_title('KITC W influential')



## Studentized Residuals Leverage Model
# There are two influential leverage points, 1 very influential
fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.influence_plot(model_4, ax=ax)

## Getting the influential points of model_4
infl = model_4.get_influence()
print(infl)
# Obs 972: studentized residual of -13.8296 (extra spicy hot!)
# Obs 399: studentized residual of -8.53774 (a little spicy hot)

## Removing observation 399 and 972 

train_final_1 = train_final.drop(train_final.index[[399, 972]])



## Re-running the regression without the two influential points

X = train_final_1[["OverallQual", "TotalBsmtSF", "GrLivArea", "KitchenQual"]]
y = train_final_1["SalePrice"]
X = sm.add_constant(X)

model_5 = sm.OLS(y,X).fit()
predictions_5 = model_5.predict(X)

model_5.summary()
# Much better adj. R2 of .819
# This model accounts for 82% of the variability in Sales Price
# Condition value may be improved by the transformations found below
# Very small AIC and SBC figures (lowest we've seen!)




## Re-running residual plots for each variable to determine if transformation needed

# Much, much better by omission, but still megaphone effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final_1.TotalBsmtSF, 
              train_final_1.SalePrice).set_title('BSMT W/out influential')

# Much better by omission, but still megaphonoe effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final_1.GrLivArea, 
              train_final_1.SalePrice).set_title('GRL W/out influential')

# Unchanged by omission of influential points, still megaphone effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final_1.KitchenQual, 
              train_final_1.SalePrice).set_title('KITC W/out influential')

# Megaphone effect
plt.subplots(figsize = (12,8))
sns.residplot(train_final_1.OverallQual, 
              train_final_1.SalePrice).set_title('OverallQual W/out influential')






#------------------------------------------------------------------------------
### Transforming the data with boxcox



## Using power transform, method = boxcox
# The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood
print(power_transform(train_final_1, method = 'box-cox'))
train_final_1_boxcox = power_transform(train_final_1, method = 'box-cox')



## Converting the new boxcox np array back into a pd dataframe
# I can't believe this worked and I am so proud of myself 
train_final_boxcox = pd.DataFrame(train_final_1_boxcox, 
                                    index=train_final_1.index,
                                    columns=train_final_1.columns)



## Running the final reg. model with transformed dataframe
X = train_final_boxcox[["OverallQual", "TotalBsmtSF", "GrLivArea", "KitchenQual"]]
y = train_final_boxcox["SalePrice"]
X = sm.add_constant(X)

model_6 = sm.OLS(y,X).fit()
predictions_6 = model_6.predict(X)

model_6.summary()
# Successfully alleviated the large condition number, yay!



#                           OLS Regression Results                            
#==============================================================================
#Dep. Variable:              SalePrice   R-squared:                       0.815
#Model:                            OLS   Adj. R-squared:                  0.814
#Method:                 Least Squares   F-statistic:                     1195.
#Date:                Tue, 16 Jun 2020   Prob (F-statistic):               0.00
#Time:                        16:43:04   Log-Likelihood:                -628.79
#No. Observations:                1092   AIC:                             1268.
#Df Residuals:                    1087   BIC:                             1293.
#Df Model:                           4                                         
#Covariance Type:            nonrobust                                         
#===============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
#-------------------------------------------------------------------------------
#const        2.478e-14      0.013    1.9e-12      1.000      -0.026       0.026
#OverallQual     0.3852      0.021     18.675      0.000       0.345       0.426
#TotalBsmtSF     0.2303      0.016     14.823      0.000       0.200       0.261
#GrLivArea       0.3218      0.017     19.049      0.000       0.289       0.355
#KitchenQual     0.1729      0.018      9.818      0.000       0.138       0.208
#==============================================================================
#Omnibus:                      315.098   Durbin-Watson:                   1.950
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1372.723
#Skew:                          -1.294   Prob(JB):                    8.26e-299
#Kurtosis:                       7.844   Cond. No.                         3.02
#==============================================================================
#
#Warnings:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.






#------------------------------------------------------------------------------
### Finalized Residual Modeling ###

## Re-running residual plots for each variable to...
#determine if transformation was successful

# - Much better
plt.subplots(figsize = (12,8))
sns.residplot(train_final_boxcox.TotalBsmtSF, 
              train_final_boxcox.SalePrice).set_title('BSMT W/out influential')

# - Much better
plt.subplots(figsize = (12,8))
sns.residplot(train_final_boxcox.GrLivArea, 
              train_final_boxcox.SalePrice).set_title('GRL W/out influential')

# - Much better
plt.subplots(figsize = (12,8))
sns.residplot(train_final_boxcox.KitchenQual, 
              train_final_boxcox.SalePrice).set_title('KITC W/out influential')

# - Much better
plt.subplots(figsize = (12,8))
sns.residplot(train_final_boxcox.OverallQual, 
              train_final_boxcox.SalePrice).set_title('OverallQual W/out influential')


