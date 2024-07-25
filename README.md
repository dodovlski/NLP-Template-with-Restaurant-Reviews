# Sentiment Analysis on Reviews using Naive Bayes

This project demonstrates a sentiment analysis pipeline on review data using Python. The pipeline includes data preprocessing, feature extraction, and training a Naive Bayes classifier to predict sentiment.

## Libraries
- `numpy`
- `pandas`
- `re`
- `nltk`
- `sklearn`

## Steps

1. **Data Loading and Preprocessing**

    ```python
    import numpy as np
    import pandas as pd

    comments = pd.read_csv('processed_rr.csv')
    ```

    The dataset `processed_rr.csv` is loaded into a pandas DataFrame.

2. **Get Rid Of Punctuation Marks**

    ```python
    import re 

    eg_comment = re.sub('[^a-zA-Z]',' ', comments['Review'][22])
    ```

    Regular expressions are used to remove non-letter characters from the text.

3. **Write All Words In Lower Case**

    ```python
    eg_comment = eg_comment.lower()
    ```

    Convert all characters in the text to lowercase.

4. **Convert To List**

    ```python
    eg_comment = eg_comment.split()
    ```

    Split the text into a list of words.

5. **Remove Meaningless Words (Stop Words)**

    ```python
    import nltk
    stops = nltk.download('stopwords')
    from nltk.corpus import stopwords
    ```

    Download and use the stop words list from the NLTK library.

6. **Find Word Origin (Stemming)**

    ```python
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    eg_comment = [ps.stem(word) for word in eg_comment if not word in set(stopwords.words('english'))]
    ```

    Stem the words using the PorterStemmer from NLTK.

7. **Restring The Last List**

    ```python
    eg_comment = ' '.join(eg_comment)
    ```

    Convert the list of words back into a single string.

8. **Apply These Steps To All Lines**

    ```python
    new_comments = []
    for i in range(716):
        _comment = re.sub('[^a-zA-Z]',' ', comments['Review'][i])
        _comment = _comment.lower()
        _comment = _comment.split()
        _comment = [ps.stem(word) for word in _comment if not word in set(stopwords.words('english'))]
        _comment = ' '.join(_comment)
        new_comments.append(_comment)
    ```

    Apply the preprocessing steps to all reviews in the dataset.

9. **Feature Extraction (Bag of Words)**

    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=2000)

    X = cv.fit_transform(new_comments).toarray()  # Independent Variable
    y = comments.iloc[:,1].values  # Dependent Variable
    ```

    Use `CountVectorizer` to convert text data into a matrix of token counts.

10. **Fill NaN Values**

    ```python
    comments.iloc[:,1] = comments.iloc[:,1].fillna(0)
    ```

    Fill any NaN values in the dependent variable column with a default value of 0.

11. **Machine Learning**

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    ```

    Split the data into training and testing sets, then train a Naive Bayes classifier and make predictions on the test set.

12. **Measuring Model Performance**

    - **Confusion Matrix**

        ```python
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        ```

    - **Accuracy**

        ```python
        accuracy = (cm[0,0] + cm[1,1]) / y_test.shape
        print(float(accuracy))
        ```

    Print the confusion matrix and calculate the accuracy of the model.

## Usage

1. Ensure you have the necessary libraries installed:

    ```bash
    pip install numpy pandas nltk scikit-learn
    ```

2. Download the NLTK stop words:

    ```python
    import nltk
    nltk.download('stopwords')
    ```

3. Place your `processed_rr.csv` file in the same directory as the script.

4. Run the script to preprocess the data, train the model, and evaluate its performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
