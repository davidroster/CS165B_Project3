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

    #TRYING TO TURN MY VALUES INTO INT FOR FP, TP, TN, FN
    # for i in range(1, len(matrix_training)):
    #     for j in range(1, len(matrix_training[i])):
    #         matrix_training[i][j] = float(matrix_training[i][j])
    #         print(type(matrix_training[i][j]))
    #         print(matrix_training[i][j])


    #TRAINING
    #Get number of rows and columns
    numrows_train = len(matrix_training)    # 3 rows in your example
    numcols_train = len(matrix_training[0]) # 2 columns in your example
    print("Number of Columns", numcols_train)
    print("Number of Rows", numrows_train)

    #Get column of just Good Movie
    #Used to compare against DTC results
    temp_GM_Vector_train = np.array(matrix_training)
    print("Original Matrix")
    print(temp_GM_Vector_train)
    GM_Vector_train = temp_GM_Vector_train[:, [numcols_train-1]]
    GM_Vector_train = GM_Vector_train[1:]
    #Changes my numpy aray from a string to an int
    GM_Vector_train = GM_Vector_train.astype(np.int)
    #GM_Vector_train = GM_Vector_train[1:]
    print("GM_Vector_train")
    print(GM_Vector_train)


    #Remove last column and first Row
    #Used to set up our X-Train for DTC Training
    trim_matrix = []

    for row in matrix_training:
        trim_matrix.append(row[:-1])

    trim_matrix = trim_matrix[1:]

    # for item in trim_matrix:
    #     trim_matrix[item].pop(0)

    NON_GM_Vector_Train = np.array(trim_matrix)
    # NON_GM_Vector_Train = NON_GM_Vector_Train[:, [1:]]


    #Trying to use Pandas because I cant remove "#" column with numpy for some reason

    print("NON_GM_Vector_Train")
    print(NON_GM_Vector_Train)



    # TESTING

    #Convert Text File into a usuable Matrix
    matrix_testing = []
    for line in testing_file.readlines():
        y = [value for value in line.split()]
        matrix_testing.append( y )

    #TRYING TO TURN MY VALUES INTO INT FOR FP, TP, TN, FN
    # for i in range(1, len(matrix_testing)):
    #     for j in range(1, len(matrix_testing[i])):
    #         matrix_testing[i][j] = float(matrix_testing[i][j])


    #Get number of rows and columns
    numrows_test = len(matrix_testing)    # 3 rows in your example
    numcols_test = len(matrix_testing[0]) # 2 columns in your example
    print("Number of Columns", numcols_test)
    print("Number of Rows", numrows_test)

    #Get column of just Good Movie
    #Used to compare against DTC results
    temp_GM_Vector_train = np.array(matrix_testing)
    print("Original Matrix")
    print(temp_GM_Vector_train)
    GM_Vector_test = temp_GM_Vector_train[:, [numcols_test-1]]
    GM_Vector_test = GM_Vector_test[1:]
    #GM_Vector_test = GM_Vector_test[1:]
    print("Y-Test")

    #Changes my numpy aray from a string to an int
    GM_Vector_test = GM_Vector_test.astype(np.int)
    print(GM_Vector_test)


    #Remove last column and first Row
    #Used to set up our X-Train for DTC Training
    trim_matrix = []

    for row in matrix_testing:
        trim_matrix.append(row[:-1])

    trim_matrix = trim_matrix[1:]

    NON_GM_Vector_Test = np.array(trim_matrix)
    print("NON_GM_Vector_Train")
    print(NON_GM_Vector_Test)



    #Decision Tree Work
    #Gini
    gini_model = tree.DecisionTreeClassifier(random_state = 0)
    gini_model.fit(NON_GM_Vector_Train, GM_Vector_train)
    gini_y_predict = gini_model.predict(NON_GM_Vector_Test)

    gini_y_predict_list = gini_y_predict.tolist()
    GM_Vector_test_list = GM_Vector_test.tolist()
    print(GM_Vector_test_list)
    print(gini_y_predict_list)
    #Gives me my accuracy which I can use to find error rate
    gini_Accuracy = accuracy_score(GM_Vector_test, gini_y_predict)
    gini_Error_Rate = (1 - gini_Accuracy)

    #entropy
    entropy_model = tree.DecisionTreeClassifier(criterion="entropy", random_state = 0)
    entropy_model.fit(NON_GM_Vector_Train, GM_Vector_train)
    entropy_y_predict = entropy_model.predict(NON_GM_Vector_Test)

    entropy_y_predict_list = entropy_y_predict.tolist()
    GM_Vector_test_list = GM_Vector_test.tolist()
    print(GM_Vector_test_list)
    print(entropy_y_predict_list)
    #Gives me my accuracy which I can use to find error rate
    entropy_Accuracy = accuracy_score(GM_Vector_test, entropy_y_predict)
    entropy_Error_Rate = (1 - entropy_Accuracy)









    # gini_gini_True_Negatives = 0
    # True_Positives = 0
    # gini_False_Negatives = 0
    # gini_False_Positives = 0

    gini_True_Positives, gini_False_Positives, gini_True_Negatives, gini_False_Negatives = Values_Calculator(GM_Vector_test_list, gini_y_predict_list)
    entropy_True_Positives, entropy_False_Positives, entropy_True_Negatives, entropy_False_Negatives = Values_Calculator(GM_Vector_test_list, entropy_y_predict_list)
    # gini_True_Positives, gini_False_Positives, gini_True_Negatives, gini_False_Negatives = Values_Calculator(GM_Vector_test_list, gini_y_predict_list)

    print("True positives:", gini_True_Positives)
    print("True negatives:", gini_True_Negatives)
    print("False positives:", gini_False_Positives)
    print("False negatives:", gini_False_Negatives)
    print("Error rate:", gini_Error_Rate)

    print("True positives:", entropy_True_Positives)
    print("True negatives:", entropy_True_Negatives)
    print("False positives:", entropy_False_Positives)
    print("False negatives:", entropy_False_Negatives)
    print("Error rate:", entropy_Error_Rate)


    #USE THIS ARTICLE



    # #Create Decision Tree classifer object
    # clf = DecisionTreeClassifier()

    # # Train Decision Tree Classifer
    # clf = clf.fit(NON_GM_Vector_Train,GM_Vector_test)

    # #Predict the response for test dataset
    # y_pred = clf.predict(GM_Vector_test)

    # print(y_pred)


    return {
            "gini":{
                'True positives':gini_True_Positives,
                'True negatives':gini_True_Negatives,
                'False positives':gini_False_Positives,
                'False negatives':gini_False_Negatives,
                'Error rate':gini_Error_Rate
                },
            "entropy":{
                'True positives':entropy_True_Positives,
                'True negatives':entropy_True_Negatives,
                'False positives':entropy_False_Positives,
                'False negatives':entropy_False_Negatives,
                'Error rate':0.00}
                }



    pass

def Values_Calculator(Y_Answer, Y_predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    print("Y_Answer")
    print(Y_Answer)

    print("Y_predicted")
    print(Y_predicted)

    for i in range(1, len(Y_predicted)):
        print("Entering loop")
        print("Y_Answer", Y_Answer[i])
        print("Y_predicted", Y_predicted[i])

        if Y_Answer[i][0]==Y_predicted[i]==1:
           TP += 1
        if Y_predicted[i]==1 and Y_Answer[i][0]!=Y_predicted[i]:
           FP += 1
        if Y_Answer[i][0]==Y_predicted[i]==0:
           TN += 1
        if Y_predicted[i]==0 and Y_Answer[i][0]!=Y_predicted[i]:
           FN += 1

    return(TP, FP, TN, FN)


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
