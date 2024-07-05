# About the Dataset.
# 1.id - unique id for a news article
# 2.title - the title of a news article
# 3.author - author of the news article
# 4.text - the text of the article(can be null)
# 5.label - a label that marks whether news article is real or fake
# 1 - Fake News
# 0 - Real News

# importing dependencies
import numpy as np
import pandas as pd
import re  # regular expression useful for searching text in a document.
from nltk.corpus import stopwords  # natural lang tool kit, corpus is the body of the text, stopword are those that
# don't add much meaning to the text eg. articles, where, what, when.
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer  # helps to convert text to feature vectors
from sklearn.model_selection import train_test_split  # splitting training data & test data
from sklearn.linear_model import LogisticRegression  # Logistic Regression Model
from sklearn.metrics import accuracy_score

# print(nltk.download('stopwords'))

# printing the stopwords in English
# print(stopwords.words('english'))  # mention which lang we are using.

# Data Pre-processing
# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('C:/Users/DELL/PycharmProjects/MLprojects/Fake News.csv')

# checking the no. of rows and columns
# print(news_dataset.shape)

# Reading the 1st five rows
print(news_dataset.head())

# counting the number of missing values in the dataset
# print(news_dataset.isnull().sum())

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title into a new column 'content'
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# print(news_dataset['content'])

# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# print(X)
# print(Y)

# Stemming:
# Stemming is the process of reducing a word to it's Root Word
# example - actor, actress, acting-> act

port_stem = PorterStemmer()  # we use this function for stemming


# we create a function stemming that will perform stemming snippet-wise on the text.
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Regular Expression searches words in a text, sub method
    # is use to replace words other than alphabetical lowercase or uppercase words from the content table with escape.
    stemmed_content = stemmed_content.lower() # convert all the words to lowercase
    stemmed_content = stemmed_content.split()  # we split all the words and convert it into a list
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  # will only store words athat are not in the stopwords list(basically removing stopwords)
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# news_dataset['content'] = news_dataset['content'].apply(stemming)  # now we apply this function to the content column
# print(news_dataset['content'])

# separating the data label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# print(X)
# print(Y)
# print(Y.shape)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()  # Tf- Term Frequency, id - inverse document frequency(basically counts the no. of words repeat in a document.
vectorizer.fit(X)

X = vectorizer.transform(X)
# print(X) # convert the text into no.s that the model can understand

# Splitting the dataset to training & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# we use stratify to divide the labels 0's and 1's into equal proportion

# Training the Model Logistic Regression

model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluation
# accuracy score of training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print('Accuracy of Training Data:',training_data_accuracy)

# accuracy score of test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print('Accuracy of Test Data:',test_data_accuracy)

# Making a Prediction System
X_new = X_test[5]  # taking values of the 4th row
prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

print(Y_test[5])  # checking if the prediction is correct against the Y_test data
