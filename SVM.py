from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def run_SVM(file_path):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :12]
    y = df.iloc[:, 12]