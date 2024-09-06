# Libraries
import numpy as np
import pandas as pd


comments = pd.read_csv('processed_rr.csv')

### Data Preprocessing

# 1) Get Rid Of Punctuation Marks
    
    # Regular Expression
import re 

    # [^a-zA-Z] -> Selects Non-Letter Characters
eg_comment = re.sub('[^a-zA-Z]',' ', comments['Review'][22])

# 2) Write All Words In Lower Case
eg_comment = eg_comment.lower()

# 3) Convert To List
eg_comment = eg_comment.split()

# 4) Remove Meaningless Words From The List ( Stop Words )
    # Natural Language Toolkit
import nltk
stops = nltk.download('stopwords')
from nltk.corpus import stopwords

# 5) Find Word Origin ( Stemming )
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

eg_comment = [ps.stem(word) for word in eg_comment if not word in set(stopwords.words('english'))]

# 6) Restring The Last List
eg_comment = ' '.join(eg_comment)

### Apply These Steps To All Lines


new_comments = []
for i in range(716):
    _comment = re.sub('[^a-zA-Z]',' ', comments['Review'][i])
    _comment = _comment.lower()
    _comment = _comment.split()
    _comment = [ps.stem(word) for word in _comment if not word in set(stopwords.words('english'))]
    _comment = ' '.join(_comment)
    new_comments.append(_comment)


### Feature Extraction ( Bag of Words )

# 7) Converting Words To Vectors 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)


X = cv.fit_transform(new_comments).toarray() # Independent Variable
y = comments.iloc[:,1].values # Dependent Variable


# 8) Fill NaN values ​​with a default value
comments.iloc[:,1] = comments.iloc[:,1].fillna(0)


### Machine Learning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)


### Measuring the success of the Naive Bayes algorithm with the Complexity Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


accuracy = (cm[0,0]+cm[1,1]) / y_test.shape

print(float(accuracy))







