# Starter code for CS 165B HW3
import numpy as np
# import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.metrics import accuracy_score

def run_train_test(training_file, testing_file):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_file: file object returned by open('training.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
    			"gini":{
    				'True positives':0,
    				'True negatives':0,
    				'False positives':0,
    				'False negatives':0,
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0,
    				'True negatives':0,
    				'False positives':0,
    				'False negatives':0,
    				'Error rate':0.00}
    				}
    """
    # TODO: IMPLEMENT HERE

    #Convert Text File into a usuable Matrix
    matrix_training = []
    for line in training_file.readlines():
        y = [value for value in line.split()]
        matrix_training.append( y )

    #TRAINING
    #Get number of rows and columns
    numrows_train = len(matrix_training)    # 3 rows in your example
    numcols_train = len(matrix_training[0]) # 2 columns in your example
    print("Number of Columns", numcols_train)
    print("Number of Rows", numrows_train)

    #Get column of just Good Movie
    #Used to compare against DTC results
    temp_y_train = np.array(matrix_training)
    print("Original Matrix")
    print(temp_y_train)
    y_train = temp_y_train[:, [numcols_train-1]]
    y_train = y_train[1:]
    #y_train = y_train[1:]
    print("Y-Train")
    print(y_train)


    #Remove last column and first Row
    #Used to set up our X-Train for DTC Training
    trim_matrix = []

    for row in matrix_training:
        trim_matrix.append(row[:-1])

    trim_matrix = trim_matrix[1:]
    X_train = np.array(trim_matrix)

    print("X-Train")
    print(X_train)



    #TESTING

    #Convert Text File into a usuable Matrix
    matrix_testing = []
    for line in testing_file.readlines():
        y = [value for value in line.split()]
        matrix_testing.append( y )

    #Get number of rows and columns
    numrows_test = len(matrix_testing)    # 3 rows in your example
    numcols_test = len(matrix_testing[0]) # 2 columns in your example
    print("Number of Columns", numcols_test)
    print("Number of Rows", numrows_test)

    #Get column of just Good Movie
    #Used to compare against DTC results
    temp_y_test = np.array(matrix_testing)
    print("Original Matrix")
    print(temp_y_test)
    y_test = temp_y_test[:, [numcols_test-1]]
    y_test = y_test[1:]
    #y_test = y_test[1:]
    print("Y-Test")
    print(y_test)


    #Remove last column and first Row
    #Used to set up our X-Train for DTC Training
    trim_matrix = []

    for row in matrix_testing:
        trim_matrix.append(row[:-1])

    trim_matrix = trim_matrix[1:]
    X_test = np.array(trim_matrix)

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    x = accuracy_score(y_test, y_predict)
    print(x)

    #USE THIS ARTICLE



    # #Create Decision Tree classifer object
    # clf = DecisionTreeClassifier()

    # # Train Decision Tree Classifer
    # clf = clf.fit(X_train,y_test)
    
    # #Predict the response for test dataset
    # y_pred = clf.predict(y_test)

    # print(y_pred)


    pass



#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()
