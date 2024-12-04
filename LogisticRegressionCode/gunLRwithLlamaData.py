import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the datasets
train_df = pd.read_csv('gun_train_data.csv')
dev_df = pd.read_csv('data/data-20241202T145651Z-001/data/gun_control_dev.csv')

# List of question columns to include for averaging (focused on the specified columns)
question_columns = [
    'Text Question 2', 'Text Question 3', 'Text Question 6',
    'Image Question 1', 'Image Question 3'
]

# Calculate the average confidence value for the selected question columns
train_df['avg_confidence'] = train_df[question_columns].mean(axis=1)
dev_df['avg_confidence'] = dev_df[question_columns].mean(axis=1)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=6000)

# Fit the vectorizer on the averaged confidence values (converted to strings) and transform both datasets
X_train = vectorizer.fit_transform(train_df['avg_confidence'].astype(str))
X_dev = vectorizer.transform(dev_df['avg_confidence'].astype(str))

# Logistic regression for the 'Support' column
clf_support = LogisticRegression()

# Train the classifier on the training data (target: 'Support')
clf_support.fit(X_train, train_df['Support'])

# Accuracy on the training data
train_accuracy_support = clf_support.score(X_train, train_df['Support'])
print(f'Support Training Accuracy: {train_accuracy_support:.4f}')

# Predictions and accuracy on the dev data (target: 'Support')
predictions_support = clf_support.predict(X_dev)
dev_accuracy_support = clf_support.score(X_dev, dev_df['Support'])
print(f'Support Dev Accuracy: {dev_accuracy_support:.4f}')
print("Support Classification Report:")
print(classification_report(dev_df['Support'], predictions_support))

# Logistic regression for the 'Persuasive' column
clf_persuasive = LogisticRegression()

# Train the classifier on the training data (target: 'Persuasive')
clf_persuasive.fit(X_train, train_df['Persuasive'])

# Accuracy on the training data
train_accuracy_persuasive = clf_persuasive.score(X_train, train_df['Persuasive'])
print(f'Persuasive Training Accuracy: {train_accuracy_persuasive:.4f}')

# Predictions and accuracy on the dev data (target: 'Persuasive')
predictions_persuasive = clf_persuasive.predict(X_dev)
dev_accuracy_persuasive = clf_persuasive.score(X_dev, dev_df['Persuasive'])
print(f'Persuasive Dev Accuracy: {dev_accuracy_persuasive:.4f}')
print("Persuasive Classification Report:")
print(classification_report(dev_df['Persuasive'], predictions_persuasive))
