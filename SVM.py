from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
    
    svm = SVC(kernel='sigmoid', decision_function_shape='ovo')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    # print(accuracy_score(y_test, y_pred))
    
    return accuracy_score(y_test, y_pred)
    
# run_SVM("gun_train_data.csv")