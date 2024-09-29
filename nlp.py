# Libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Data Loading
comments = pd.read_csv('processed_rr.csv')

# Data Preprocessing

# 1) Get Rid Of Punctuation Marks
eg_comment = re.sub('[^a-zA-Z]', ' ', comments['Review'][22])

# 2) Convert to Lowercase
eg_comment = eg_comment.lower()

# 3) Convert to List
eg_comment = eg_comment.split()

# 4) Remove Stop Words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 5) Stemming (Finding the Root of Words)
ps = PorterStemmer()
eg_comment = [ps.stem(word) for word in eg_comment if word not in stop_words]

# 6) Re-string the Last List
eg_comment = ' '.join(eg_comment)

# Apply These Steps to All Reviews

new_comments = []
for i in range(len(comments)):
    _comment = re.sub('[^a-zA-Z]', ' ', comments['Review'][i])
    _comment = _comment.lower()
    _comment = _comment.split()
    _comment = [ps.stem(word) for word in _comment if word not in stop_words]
    _comment = ' '.join(_comment)
    new_comments.append(_comment)

# Feature Extraction (Bag of Words)

# 7) Convert Words to Vectors
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(new_comments).toarray()  # Independent Variable

# Extract Dependent Variable (y)
y = comments.iloc[:, 1].values

# 8) Fill NaN Values with a Default Value
comments.iloc[:, 1] = comments.iloc[:, 1].fillna(0)

# Machine Learning

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train Naive Bayes Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on Test Data
y_pred = gnb.predict(X_test)

# Evaluate the Model Using Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate Accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / y_test.shape[0]
print(f"Accuracy: {accuracy:.4f}")
