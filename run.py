import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pdb
from operator import add
from sklearn.model_selection import KFold # import KFold
import sys



def calculateMseCost(X, Y, theta):
    """Given the input data and the actual output, and the theta, calculates the Mean Squared Error
        between the predictions and the actual values. Predictions are calculated by performin a dot product
        between X and theta.

        Keyword arguments:
        X -- A data frame, representing the input data set that has the data for which we will perform the prediction.
        Y -- A series, that has the actual values for every element in the data set.
        theta -- The coefficients that are used to calculate the predictions.
    """
    count = len(X)        
    X_transformed = X.copy()
    X_transformed.insert(0, 'for theta 0', 1)
    prediction = np.dot(X_transformed.values, theta)
    difference = np.square(prediction - Y.values)
    cost = (1/1)*(1/count) * np.sum(difference)
    return cost

def gradient_descent_matrix(X_train, Y_train, alpha):    
    """Performs Batch Gradient descent for Linear Regression, by employing the matrix approach.
        The algorithm continues to take steps until either the difference in theta becomes less than 0.000001, or
        1 million iterations have been performed.

        Keyword Arguments:
        X_train : A data frame that has the training data.
        Y_train: A data frame that has the expected output for the training data.
        alpha: The learning rate.
    """
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
    """Performs a single step of gradient descent using the iterative approach.
        Calculates the partial derivatives for both thetaZero and thetaOne, and then returns the updated thetaZero 
        and thetaOne values.

        Keyword Arguments:
        X_train: The input training data with the features and their values.
        Y_train: The expected output for the training data.
        thetaZeroStart: The first coefficient
        thetaOneStart: The second coefficient
        alpha: The learning rate.
    """
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

def iterative_gradient_descent(X_train, Y_train, thetaZero, thetaOne, alpha):
    """Performs the iterative gradient descent for  linear regression.
        After performing multiple iterations, this method returns the coefficients for the 
        linear regression model for the given training data.

        Keyword Arguments:
        X_train: The input training data
        Y_train: The expected output for the input training data.
        thetaZero: The first coefficient (this is the intercept for the linear regression model.)
        thetaOne : The second coefficient for linear regression.

    """
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
    """Performs Linear Regression using the sklearn package.
        Returns the coefficients for the linear regression model after fitting the input training data.

        Keyword Arguments:
        X_train: The input training data
        Y_train: The expected output for the input training data.
        X_test: The test data with the features
        y_test: The test data with the actual ouput values.
        feature_cols: The actual values
    """
    # Fitting the model
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print("\nCoefficients:", list(zip(feature_cols, linreg.coef_)))
    print("Intercept:", linreg.intercept_)
    y_pred = linreg.predict(X_test)
    #print("\nMAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE from sklearn", metrics.mean_squared_error(y_test, y_pred))

    return [linreg.intercept_, linreg.coef_[0]]


def plot_data(x, y):
    """Plots the data as a scatter plot. Also both the inputs must be data frames.
    """
    plt.scatter(x, y)
    plt.show() # Depending on whether you use IPython or interactive mode, etc.
    data.plot(style=['o','rx'])


def main(filename):
    print("Reading the data..")
    data = pd.read_csv(filename)
    print("Finished reading the data..")
    # Preparing the train and test data
    feature_cols = ['X']
    X = data[feature_cols]
    y = data['Y']
    print("Splitting the data using k fold approach, with k = 3")


    foldCount = 3

    # To perform cross validation.

    kf = KFold(n_splits=foldCount)

    averageMseIterative = 0
    averageMseMatrix = 0
    averageMseSkLearn = 0

    averageThetaIterative = np.array([0,0])
    averageThetaMatrix = np.array([0,0])
    averageThetaSkLearn = np.array([0,0])

    iteration = 0
    for train_index, test_index in kf.split(X):
        print("-------------------------------------------------------------------------------------------------------")
        print("\n")
        print("At iteration {} of testing.".format(iteration))
        iteration += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        thetaZeroStart = 0
        thetaOneStart = 0
        alpha = 0.001
        iterationCount = 100000

        # 3 different ways of calculating our results.

        thetaFromIterative = iterative_gradient_descent(X_train, y_train, thetaZeroStart, thetaOneStart, alpha)
        averageThetaIterative =  averageThetaIterative + np.array(thetaFromIterative)

        thetaFromMatrix = gradient_descent_matrix(X_train, y_train, alpha)
        averageThetaMatrix = averageThetaMatrix + np.array(thetaFromMatrix)

        thetaFromSkitLearn = calculateSkLearnGradientDescent(X_train, y_train, X_test, y_test, feature_cols)
        averageThetaSkLearn = averageThetaSkLearn + np.array(thetaFromSkitLearn)

        print("Theta from skit learn approach = {}".format(thetaFromSkitLearn, type(thetaFromSkitLearn)))
        print("Theta from iterative approach = {}".format(thetaFromIterative, type(thetaFromIterative)))
        print("Theta from matrix approach = {}".format(thetaFromMatrix, type(thetaFromMatrix)))
        print("\n")

        cost = calculateMseCost(X_test, y_test, np.array(thetaFromSkitLearn))
        print("MSE Cost using Sk Learn = {}".format(cost))
        averageMseSkLearn += cost

        cost = calculateMseCost(X_test, y_test, np.array(thetaFromIterative))
        print("MSE Cost from Iterative Approach = {}".format(cost))
        averageMseIterative += cost

        cost = calculateMseCost(X_test, y_test, np.array(thetaFromMatrix))
        print("MSE Cost from matrix approach  = {}".format(cost))
        averageMseMatrix += cost

    # Calculate the average.

    averageMseIterative /= 3
    averageMseSkLearn /= 3
    averageMseMatrix /= 3

    averageThetaIterative  = averageThetaIterative/3
    averageThetaSkLearn = averageThetaSkLearn/3
    averageThetaMatrix = averageThetaMatrix/3

    print("---------------------------------------------------------------------------------------------------------")
    print("Final Results...\n\n")
    print("Theta:")
    print("Theta SkLearn: {}".format(averageThetaSkLearn))
    print("Theta Matrix: {}".format(averageThetaMatrix))
    print("Theta Iterative: {}".format(averageThetaIterative))
    print("\n")
    print("MSE")
    print("MSE SkLearn: {}".format(averageMseSkLearn))
    print("MSE Matrix: {}".format(averageMseMatrix))
    print("MSE Iterative: {}".format(averageMseIterative))    
    print("Program complete..")


if __name__ == '__main__':    
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    filename = sys.argv[1]
    main(filename)











