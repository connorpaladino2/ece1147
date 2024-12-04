from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def run_SVM(file_path):
    df = pd.read_csv(file_path)

    X = df.iloc[1:, :12]
    y = df.iloc[1:, 12]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    svm = SVC(kernel='sigmoid', decision_function_shape='ovo')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    # print(accuracy_score(y_test, y_pred))
    
    return accuracy_score(y_test, y_pred)
    
# run_SVM("gun_train_data.csv")