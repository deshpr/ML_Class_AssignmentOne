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

def calculateCost(X, Y, theta):
    count = len(X)        
    output = X.dot(theta)
    cost =  (1/2*count) * np.sum(np.square(output.values  - Y.values.reshape(len(Y), 1)))
    return cost

"""
def linear_regression_gradient_descent(X_train, Y_train):
    theta  = np.random.randn(2,1)
    iterationCount = 150
    alpha = 0.01
    theta, thetaSoFar, costsSoFar = batch_gradient_descent(X_train, Y_train, theta, alpha, iterationCount)
    print("complete")
"""


def calculateCost(thetaZero, thetaOne, X, Y):
    total = 0
    for i in range(0, len(X)):
#        pdb.set_trace()
        yValue = Y.iloc[i]
        xValue = X.iloc[i,0]
        prediction = xValue * thetaOne + thetaZero
        error = prediction - yValue
        total += (error * error)
    return total/(2 * len(X))





def gradient_descent(X_train, Y_train, thetaZero, thetaOne, alpha, iterationCount = 20):
    i = -1
    while True:
        i += 1
        print("At iteration = {}".format(i))
        thetaZeroNew, thetaOneNew =  step_gradient_descent(X_train, Y_train, thetaZero, thetaOne, alpha)
        thetaZeroChange = abs(thetaZeroNew - thetaZero)
        thetaOneChange = abs(thetaOneNew - thetaOne)
        thetaZero = thetaZeroNew
        thetaOne = thetaOneNew
        print("theta zero = {}, theta one = {}".format(thetaZero, thetaOne))
        if (thetaZeroChange <= 0.0001  and thetaOneChange <= 0.0001) or (i >= 1000000):
            print("ending at iteration = {}".format(i))
            break
    return thetaZero, thetaOne

def step_gradient_descent(X_train, Y_train, thetaZeroStart, thetaOneStart, alpha):
    # Add it in the beginning
    count = len(X_train)
    thetaZeroGradient = 0
    thetaOneGradient = 0
#    X_train_Transformed.insert(0, 'for theta 0', 1)
    for i in range(0, len(X_train)):
#        pdb.set_trace()
        yValue = Y_train.iloc[i]
        xValue = X_train.iloc[i,0]
#        print("x = {}, y  = {}".format(xValue, yValue))
#        thetaZero_Gradient += -(2/count) * (yValue - (thetaOneStart * xValue + thetaZeroStart))
#       thetaOne_Gradient += -(2/count) * xValue * (yValue - (thetaOneStart * xValue + thetaZeroStart))
        thetaZeroGradient += (thetaOneStart * xValue) + thetaZeroStart - yValue
        thetaOneGradient += ((thetaOneStart * xValue) + thetaZeroStart - yValue)*xValue
    print("theta Zero gradient = {}, theta one gradient = {}".format(thetaZeroGradient, thetaOneGradient))
    newThetaZero = thetaZeroStart - (alpha * (1/count) * thetaZeroGradient)
    newThetaOne = thetaOneStart - (alpha * (1/count) * thetaOneGradient) 
    return [newThetaZero, newThetaOne]


    """
    costsSoFar = np.zeros(iterationCount)
    thetaSoFar = np.zeros((iterationCount,2))
    for iteration in range(iterationCount):
        print("Start iteration = {}".format(iteration))
        predictionOutput = X_train_Transformed.dot(theta)
        thetaBefore = theta.copy()
        print("prediction outout calculated")
        predictionOutputSeries = predictionOutput.iloc[:,0]
        difference = (predictionOutputSeries - Y_train).values
        rightHandSide  = (1/count)*alpha*(X_train_Transformed.T.dot(difference))
        theta = theta - rightHandSide.values.reshape(len(rightHandSide.values), 1)
        print("Theta = {}, theta before = {}".format(theta, thetaBefore))        
        thetaSoFar[iteration,:] = theta.T
        costsSoFar[iteration] = calculateCost(X_train_Transformed, Y_train, theta)
        if iteration > 0:
            difference = (costsSoFar[iteration] - costsSoFar[iteration - 1])
        else:
            difference = costsSoFar[iteration]
        print("differencde in cost = {}".format(difference))
        if iteration > 0 and difference <= 0.001:
            print("done..")
            break
    pdb.set_trace()
    """
    return theta, thetaSoFar, costsSoFar



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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print("calculate coefficients")
#thetaZero, thetaOne = calculate_coefficients(X_train, y_train)
#print("theta Zeor = {}, theta One = {}".format(thetaZero, thetaOne))

#linear_regression(X_train, y_train)
thetaZeroStart = 0
thetaOneStart = 0
alpha = 0.001
iterationCount = 100000
thetaZero, thetaOne = gradient_descent(X_train, y_train, thetaZeroStart, thetaOneStart, alpha, iterationCount)
print("theta Zeor = {}, theta One = {}".format(thetaZero, thetaOne))



"""
# Data
plt.scatter(data['X'], data['Y'])
plt.show() # Depending on whether you use IPython or interactive mode, etc.
data.plot(style=['o','rx'])

"""


# Fitting the model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print("\nCoefficients:", list(zip(feature_cols, linreg.coef_)))
print("Intercept:", linreg.intercept_)

# Evaluating the model
y_pred = linreg.predict(X_test)
print("\nMAE:", metrics.mean_absolute_error(y_test, y_pred))
#print("MSE:", metrics.mean_squared_error(y_test, y_pred))
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




