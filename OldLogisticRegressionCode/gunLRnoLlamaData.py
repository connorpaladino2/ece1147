import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_df = pd.read_csv('data/data-20241202T145651Z-001/data/gun_control_train.csv')
dev_df = pd.read_csv('data/data-20241202T145651Z-001/data/gun_control_dev.csv')

vectorizer = TfidfVectorizer(max_features=6000)

#fit the vectorizer on the training data and transform both training and dev data
X_train = vectorizer.fit_transform(train_df['tweet_text'])
X_dev = vectorizer.transform(dev_df['tweet_text'])

clf = LogisticRegression()

#train the classifier on the training data
clf.fit(X_train, train_df['stance'])

#check the accuracy on the training data
train_accuracy = clf.score(X_train, train_df['stance'])
print(f'Training Accuracy: {train_accuracy:.4f}')

#make predictions on the dev data
predictions = clf.predict(X_dev)

#accuracy on the dev data
dev_accuracy = clf.score(X_dev, dev_df['stance'])
print(f'Dev Accuracy: {dev_accuracy:.4f}')
print(classification_report(dev_df['stance'], predictions))
