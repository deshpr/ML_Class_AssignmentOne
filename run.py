import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pdb

def calculate_coefficients(X_train, Y_train):
    # get data as a series
    X_train_series = X_train[X_train.columns[0]]
    print("input type, x = {}, y  = {}".format(type(X_train_series), type(Y_train)))
    mean_x = X_train_series.mean()
    mean_y = Y_train.mean()
    print("mean x = {}, mean y = {}".format(mean_x, mean_y))
    mean_product = mean_x * mean_y
    print("sum")
    sum_product = (Y_train * X_train_series).sum()
    count = len(X_train_series)

    X_Mean_Squared = mean_x * mean_x
    X_Squared_Sum = (X_train_series ** 2).sum()

    Coefficient_One = sum_product - (count * mean_x * mean_y)
    Coefficient_Two = X_Squared_Sum - (count * X_Mean_Squared)

    thetaOne = Coefficient_One/Coefficient_Two
    thetaZero = mean_y - (thetaOne*mean_x)

    return thetaZero, thetaOne

def linear_regression(X_train, Y_train):
    thetaZero, thetaOne = calculate_coefficients(X_train, Y_train)
    Y_prediction =  thetaZero + (thetaOne * X_train)
    # plotting the regression line 
    plt.plot(X_train, Y_prediction, color = "g") 
    plt.scatter(X_train, Y_train)
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
    # function to show plot 
    plt.show() 

def calculateMseCost(X, Y, theta):
    count = len(X)        
    X_transformed = X.copy()
    X_transformed.insert(0, 'for theta 0', 1)
    prediction = np.dot(X_transformed.values, theta)
    difference = np.square(prediction - Y.values)
    cost = (1/1)*(1/count) * np.sum(difference)
    return cost


def calculateCostIterative(thetaZero, thetaOne, X, Y):
    total = 0
    for i in range(0, len(X)):
#        pdb.set_trace()
        yValue = Y.iloc[i]
        xValue = X.iloc[i,0]
        prediction = xValue * thetaOne + thetaZero
        error = prediction - yValue
        total += (error * error)
    return total/(2 * len(X))

def getPredictions(X, theta):
    return np.dot(X, theta)

def gradient_descent_matrix(X_train, Y_train, alpha, iterationCount):    
    X_train_transformed = X_train.copy()
    X_train_transformed.insert(0, 'for theta 0', 1)
    featureCount = len(X_train_transformed.columns)
    zeros = [0] * featureCount
    stepCount = -1
    thetaInitial = np.array(zeros)
    count = len(X_train)
    while True:
        stepCount += 1
        difference = (np.dot(X_train_transformed.values, thetaInitial) - Y_train.values)
        thetaUpdate = thetaInitial - alpha * (1/count) * (np.dot(X_train_transformed.T.values, difference))
        thetaDifference = np.absolute(thetaUpdate - thetaInitial)
        maxElement = max(thetaDifference)        
        thetaInitial  = thetaUpdate
        if maxElement < 0.000001:
            print("ending at iteration at stepCount = {} ".format(stepCount))
            break
        if stepCount > 1000000:
            print("ending at iteration at stepCount = {} ".format(stepCount))        
        
    print("Theta = {}".format(thetaInitial))
    return thetaInitial


def iterative_step_gradient_descent(X_train, Y_train, thetaZeroStart, thetaOneStart, alpha):
    # Add it in the beginning
    count = len(X_train)
    thetaZeroGradient = 0
    thetaOneGradient = 0
    for i in range(0, len(X_train)):
        yValue = Y_train.iloc[i]
        xValue = X_train.iloc[i,0]
        thetaZeroGradient += (thetaOneStart * xValue) + thetaZeroStart - yValue
        thetaOneGradient += ((thetaOneStart * xValue) + thetaZeroStart - yValue)*xValue
#    print("theta Zero gradient = {}, theta one gradient = {}".format(thetaZeroGradient, thetaOneGradient))
    newThetaZero = thetaZeroStart - (alpha * (1/count) * thetaZeroGradient)
    newThetaOne = thetaOneStart - (alpha * (1/count) * thetaOneGradient) 
    return [newThetaZero, newThetaOne]

def gradient_descent(X_train, Y_train, thetaZero, thetaOne, alpha, iterationCount):
    i = -1
    while True:
        i += 1
        thetaZeroNew, thetaOneNew =  iterative_step_gradient_descent(X_train, Y_train, thetaZero, thetaOne, alpha)
        thetaZeroChange = abs(thetaZeroNew - thetaZero)
        thetaOneChange = abs(thetaOneNew - thetaOne)
        thetaZero = thetaZeroNew
        thetaOne = thetaOneNew
        if (thetaZeroChange <= 0.000001  and thetaOneChange <= 0.000001) or (i >= 1000000):
            print("ending at iteration = {}".format(i))
            break
    return [thetaZero, thetaOne]

def calculateSkLearnGradientDescent(X_train, y_train, X_test, y_test, feature_cols):
    # Fitting the model
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print("\nCoefficients:", list(zip(feature_cols, linreg.coef_)))
    print("Intercept:", linreg.intercept_)
    y_pred = linreg.predict(X_test)
    #print("\nMAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE from sklearn", metrics.mean_squared_error(y_test, y_pred))

    return [linreg.intercept_, linreg.coef_[0]]

def mean_squared_error(y_actual,  y_test):
    difference = y_actual - y_test
    squared = difference**2
    length = len(squared)
    print("length = {}".format(length))
    mean = squared.sum()/length
    print("mean  = {}".format(mean))
    return mean


filename = "data/insurance.csv"
data = pd.read_csv(filename)
print("finished reading the data")
print(data.columns.values)
# Preparing the train and test data
feature_cols = ['X']
X = data[feature_cols]
y = data['Y']
print("type, x = {}, y = {}".format(type(X), type(y)))
print("splitting the data")
from sklearn.model_selection import KFold # import KFold
foldCount = 3
kf = KFold(n_splits=foldCount)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

thetaZeroStart = 0
thetaOneStart = 0
alpha = 0.001
iterationCount = 100000

# 3 different ways of calculating our results.

thetaFromIterative = gradient_descent(X_train, y_train, thetaZeroStart, thetaOneStart, alpha, iterationCount)
thetaFromMatrix = gradient_descent_matrix(X_train, y_train, alpha, iterationCount)
thetaFromSkitLearn = calculateSkLearnGradientDescent(X_train, y_train, X_test, y_test, feature_cols)

print("Theta from skit learn approach = {}, type = {}".format(thetaFromSkitLearn, type(thetaFromSkitLearn)))
print("Theta from iterative approach = {}, type = {}".format(thetaFromIterative, type(thetaFromIterative)))
print("Theta from matrix approach = {}, type = {}".format(thetaFromMatrix, type(thetaFromMatrix)))


print("Test Theta from skit learn")
cost = calculateMseCost(X_test, y_test, np.array(thetaFromSkitLearn))
print("cost  = {}".format(cost))

print("Test Theta from iterative")
cost = calculateMseCost(X_test, y_test, np.array(thetaFromIterative))
print("cost  = {}".format(cost))
print("Test Theta from matrix approach")
cost = calculateMseCost(X_test, y_test, np.array(thetaFromMatrix))
print("cost  = {}".format(cost))



"""
# Data
plt.scatter(data['X'], data['Y'])
plt.show() # Depending on whether you use IPython or interactive mode, etc.
data.plot(style=['o','rx'])

"""


# Evaluating the model
#print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print("Length = {}, {}".format(len(y_pred), len(y_test)))
"""

mse = mean_squared_error(y_test, y_pred)
print("mse = {}".format(mse))



print("##########################################################")
print("POLYNOMIAL REGRESSION!!!!!")
poly=PolynomialFeatures(degree=4)
poly_X=poly.fit_transform(X)
# create a Linear Regressor   
lin_regressor = LinearRegression()
# fit this to Linear Regressor
lin_regressor.fit(X_train,y_train)
y_preds = lin_regressor.predict(X_test)
print("\nMAE:", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
"""




