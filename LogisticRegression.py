from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def run_LR(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[1:, :12]
    y_train = train_data.iloc[1:, 12]
    
    X_test = test_data.iloc[1:, :12]
    y_test = test_data.iloc[1:, 12]
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    
    print("Training Score: ", train_score)
    
    y_pred = lr.predict(X_test)
    
    false_positive_rate = np.sum(np.logical_and(y_test == 0, y_pred == 1))
    false_negative_rate = np.sum(np.logical_and(y_test == 1, y_pred == 0))
    
    total_accuracy = accuracy_score(y_test, y_pred)
    return total_accuracy, false_positive_rate, false_negative_rate
