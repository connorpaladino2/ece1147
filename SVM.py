from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def run_SVM(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[1:, :12]
    y_train = train_data.iloc[1:, 12]
    
    X_test = test_data.iloc[1:, :12]
    y_test = test_data.iloc[1:, 12]
    
    svm = SVC(kernel='linear', decision_function_shape='ovo')
    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    
    false_positive_rate = np.sum(np.logical_and(y_test == 0, y_pred == 1))
    false_negative_rate = np.sum(np.logical_and(y_test == 1, y_pred == 0))
    
    total_accuracy = accuracy_score(y_test, y_pred)
    
    return total_accuracy, false_positive_rate, false_negative_rate
    
run_SVM("gun_train_data.csv", "gun_test_data.csv")