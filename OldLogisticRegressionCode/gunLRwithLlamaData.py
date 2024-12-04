import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_df = pd.read_csv('gun_train_data.csv')
dev_df = pd.read_csv('gun_test_data.csv')

question_columns = [
    'Text Question 2', 'Text Question 3', 'Text Question 6',
    'Image Question 1', 'Image Question 3'
]

train_df['avg_confidence'] = train_df[question_columns].mean(axis=1)
dev_df['avg_confidence'] = dev_df[question_columns].mean(axis=1)

vectorizer = TfidfVectorizer(max_features=6000)

X_train = vectorizer.fit_transform(train_df['avg_confidence'].astype(str))
X_dev = vectorizer.transform(dev_df['avg_confidence'].astype(str))

clf_support = LogisticRegression()
clf_support.fit(X_train, train_df['Support?'])

train_accuracy_support = clf_support.score(X_train, train_df['Support?'])
print(f'Support Training Accuracy: {train_accuracy_support:.4f}')

predictions_support = clf_support.predict(X_dev)
dev_accuracy_support = clf_support.score(X_dev, dev_df['Support?'])
print(f'Support Dev Accuracy: {dev_accuracy_support:.4f}')
print("Support Classification Report:")
print(classification_report(dev_df['Support?'], predictions_support))

clf_persuasive = LogisticRegression()
clf_persuasive.fit(X_train, train_df['Pursuasive?'])

train_accuracy_persuasive = clf_persuasive.score(X_train, train_df['Pursuasive?'])
print(f'Persuasive Training Accuracy: {train_accuracy_persuasive:.4f}')

predictions_persuasive = clf_persuasive.predict(X_dev)
dev_accuracy_persuasive = clf_persuasive.score(X_dev, dev_df['Pursuasive?'])
print(f'Persuasive Dev Accuracy: {dev_accuracy_persuasive:.4f}')
print("Persuasive Classification Report:")
print(classification_report(dev_df['Pursuasive?'], predictions_persuasive))
