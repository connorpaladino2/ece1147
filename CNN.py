import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def run_CNN(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[1:, :12].values
    y_train = train_data.iloc[1:, 12].values
    X_test = test_data.iloc[1:, :12].values
    y_test = test_data.iloc[1:, 12].values

    X_train = X_train.astype('float32') / np.max(X_train, axis=0)
    X_test = X_test.astype('float32') / np.max(X_test, axis=0)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Evaluate on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype('int32').flatten()

    false_positive_rate = np.sum(np.logical_and(y_test == 0, y_pred == 1))
    false_negative_rate = np.sum(np.logical_and(y_test == 1, y_pred == 0))
    total_accuracy = accuracy_score(y_test, y_pred)

    return total_accuracy, false_positive_rate, false_negative_rate


def run_CNN_persuasion(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[1:, :12].values
    y_train = train_data.iloc[1:, 13].values
    X_test = test_data.iloc[1:, :12].values
    y_test = test_data.iloc[1:, 13].values

    X_train = X_train.astype('float32') / np.max(X_train, axis=0)
    X_test = X_test.astype('float32') / np.max(X_test, axis=0)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Evaluate on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype('int32').flatten()

    false_positive_rate = np.sum(np.logical_and(y_test == 0, y_pred == 1))
    false_negative_rate = np.sum(np.logical_and(y_test == 1, y_pred == 0))
    total_accuracy = accuracy_score(y_test, y_pred)

    return total_accuracy, false_positive_rate, false_negative_rate