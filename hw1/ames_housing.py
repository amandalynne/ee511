# coding: utf-8

# Amandalynne Paullada, EE511 Win 2018
# HW 1
# Code exported from IPYNB.

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Columns by type of variable
numerical_variables = ['Lot Area', 'Lot Frontage', 'Year Built',
                       'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
                       'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
                       '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
                       'Garage Area', 'Wood Deck SF', 'Open Porch SF', 
                       'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
                       'Pool Area']
discrete_variables = ['MS SubClass', 'MS Zoning', 'Street',
                      'Alley', 'Lot Shape', 'Land Contour',
                      'Utilities', 'Lot Config', 'Land Slope',
                      'Neighborhood', 'Condition 1', 'Condition 2',
                      'Bldg Type', 'House Style', 'Overall Qual',
                      'Overall Cond', 'Roof Style', 'Roof Matl',
                      'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
                      'Exter Qual', 'Exter Cond', 'Foundation', 
                      'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
                      'BsmtFin Type 1', 'Heating', 'Heating QC',
                      'Central Air', 'Electrical', 'Bsmt Full Bath', 
                      'Bsmt Half Bath', 'Full Bath', 'Half Bath', 
                      'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual',
                      'TotRms AbvGrd', 'Functional', 'Fireplaces',
                      'Fireplace Qu', 'Garage Type', 'Garage Cars',
                      'Garage Qual', 'Garage Cond', 'Paved Drive',
                      'Pool QC', 'Fence', 'Sale Type', 'Sale Condition']

# For convenience, store the names of metadata column & the target variable column.
meta_variables = ['Order', 'SalePrice']


# Read the housing data file as a tab-separated file

# Many fields have 'NA' as a meaningful value, so don't read that as NaN
data = pd.read_csv('AmesHousing.txt', sep='\t', keep_default_na=False, na_values=[""])


# Replace NaNs with 0 for numerical data
data[numerical_variables] = data[numerical_variables].fillna(value=0)


# Replace missing categorical feature values with "NO DATA"
data[discrete_variables] = data[discrete_variables].fillna(value="NO DATA")


# Only keep the attributes relevant to the homework (slice the dataframe)
data = data[numerical_variables+discrete_variables+meta_variables]


# Splitting the data into train, test, validation
validation = data.loc[data['Order'] % 5 ==3]
test = data.loc[data['Order'] % 5 ==4]
train = data.loc[(data['Order'] % 5 != 3) & (data['Order'] % 5 != 4)]


# Just verifying correct number of rows
# len(validation) + len(test) + len(train)


# Question 4
# One feature
X = train['Gr Liv Area'].values.reshape(-1, 1)


# Target
y = train['SalePrice'].values.reshape(-1, 1)

# Checking dimensions
# X.shape
# y.shape


# Question 4: one variable least squares linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)

y_pred = regr.predict(X)


# Show the coefficients and intercept
# print('Coefficients: \n', regr.coef_)
# print('Intercept: \n', regr.intercept_)


# Plot the data.

# plt.scatter(X, y, color='black', s=1)
# plt.plot(X, y_pred, color='blue', linewidth=2)
# 
# plt.suptitle('Ames Housing Prices ($) by Gr Liv Area (Sq Ft)')
# plt.xlabel('Sale Price')
# plt.ylabel('Gr Liv Area')
# 
# plt.show()


# Apply model to validation and report RMSE
validation_x = validation['Gr Liv Area'].values.reshape(-1, 1)
validation_true = validation['SalePrice'].values.reshape(-1, 1)

validation_pred = regr.predict(validation_x)
rms = sqrt(mean_squared_error(validation_true, validation_pred))
# print(rms)


# Question 5
# One hot encoding for categorical features
one_hot = pd.get_dummies(data, columns=discrete_variables)


# Re-split the data using the one-hot dataframe
# Remove the Order and SalePrice columns (irrelevant / cheating)
# 'ohe' = one-hot encoded
ohe_validation = (one_hot.loc[data['Order'] % 5 ==3]).drop(['Order', 'SalePrice'], axis=1)
ohe_test = (one_hot.loc[data['Order'] % 5 ==4]).drop(['Order', 'SalePrice'], axis=1)
ohe_train = (one_hot.loc[(data['Order'] % 5 != 3) & (data['Order'] % 5 != 4)]).drop(['Order', 'SalePrice'], axis=1)


# Question 5: all features linear regression model
regr = linear_model.LinearRegression()
regr.fit(ohe_train, y)

y_pred = regr.predict(ohe_validation)
rms = sqrt(mean_squared_error(validation_true, y_pred))
# print(rms)


# Question 6
# Normalize the data: subtract mean and divide by stdev. based on training data
scaler = StandardScaler()
scaler.fit(ohe_train)


# Question 6: Lasso
train_nonzeros = []

# Lasso on train
for alpha in range(50,550,50):
    lasso = linear_model.Lasso(alpha=alpha)
    scaled = scaler.transform(ohe_train)
    lasso.fit(scaled, y)
    train_pred = lasso.predict(scaled)
    rms = sqrt(mean_squared_error(y, train_pred))
    # print("Alpha: {0}".format(alpha))
    # print("RMSE {0}".format(rms))
    # print("Number of non-zero coefficients: {0}".format(np.count_nonzero(lasso.coef_)))
    train_nonzeros.append(np.count_nonzero(lasso.coef_))
    
# plt.show()

v_nonzeros = []

# Lasso on validation
for alpha in range(50,550,50):
    lasso = linear_model.Lasso(alpha=alpha)
    scaled = scaler.transform(ohe_train)
    lasso.fit(scaled, y)
    scaled_v = scaler.transform(ohe_validation)
    y_pred = lasso.predict(scaled_v)
    rms = sqrt(mean_squared_error(validation_true, y_pred))
    # print("Alpha: {0}".format(alpha))
    # print("RMSE {0}".format(rms))
    # print("Number of non-zero coefficients: {0}".format(np.count_nonzero(lasso.coef_)))
    v_nonzeros.append(np.count_nonzero(lasso.coef_))
    
# The lines are the same for both...
# plt.plot(np.asarray(range(50,550,50)), np.asarray(train_nonzeros))
# plt.plot(np.asarray(range(50,550,50)), np.asarray(v_nonzeros))
# plt.xlabel('Alpha')
# plt.ylabel('Number of nonzero coefficients')

# plt.show()


# Question 7: on test data
test_true = test['SalePrice'].values.reshape(-1, 1)


# single var
regr = linear_model.LinearRegression()
regr.fit(X, y)
test_x = test['Gr Liv Area'].values.reshape(-1, 1)
test_pred = regr.predict(test_x)
rms = sqrt(mean_squared_error(test_true, test_pred))
# print(rms)


# all var
regr = linear_model.LinearRegression()
regr.fit(ohe_train, y)

test_pred = regr.predict(ohe_test)
rms = sqrt(mean_squared_error(test_true, test_pred))
# print(rms)


# regularized
lasso = linear_model.Lasso(alpha=450)
scaled = scaler.transform(ohe_train)
lasso.fit(scaled, y)
scaled_t = scaler.transform(ohe_test)
y_pred = lasso.predict(scaled_t)
rms = sqrt(mean_squared_error(test_true, y_pred))
# print(rms)
