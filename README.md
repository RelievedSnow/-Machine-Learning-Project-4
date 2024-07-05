# -Machine-Learning-Project-4
# Fake News Prediction
# Dataset Link: https://www.kaggle.com/c/fake-news/data?select=train.csv

# Step 1: Importing Dependencies
* import numpy for numpy arrays.
* import pandas for reading '.csv' file.
* import stopwords from nltk.corpus library - natural lang tool kit, corpus is the body of the text, stopword are those that don't add much meaning to the text eg. articles, where, what, when.
* import TfidfVectorizer from sklearn.feature_extraction.text- helps to convert text to feature vectors
* import train_test_split from sklearn.model_selection for splitting the data into test data and training data.
* import Logistic Regression Model from sklearn.linear_model- We are using Logistics Pregression for prediction.
* import accuracy_score from sklearn.metricsfor evaluating the accuracy of the model.

# Step 2: Download the Stopwords.
* Using nltk.download we download the list of 'stopwords'.
* After downloading the list of 'stopwords' we mention the language we are using(english.

# Step 3: Data Collection And Pre-processing.
* We start with loading the dataset to a pandas DataFrame using the '.read_csv()' functiona and storing it into the 'news_dataset' variable.
* Checking the No. of rows and columns using the '.shape' method.
* Reading the 1st five rows using the '.head()' function.
* Counting the number of missing values in the dataset using the '.isnull()' function and '.sum()' function that counts all the missing mull values.
* We replace the null values with an empty string using '.fillna()'.
* Merging the author name and news title into a new column 'content'
* Separating the data(numerical values) & label.

# Step 4: Stemming:
* Stemming is the process of reducing a word to it's Root Word. example - actor, actress, acting-> act.
* We use PorterStemmer() function for stemming.
* We create a function stemming that will perform stemming snippet-wise on the text.

# Step 5: Seperating Text Data And Label Data.
* We seperate the Text Data and Labelled Data.

# Step 6: Converting the Textual Data to Numerical Data.
* TfidfVectorizer()- Tf- Term Frequency, id - inverse document frequency(basically counts the no. of words repeat in a document.
* Now we train the Data using the '.fit()' function.

# Step 7: Splitting the Dataset to Training & Test Data.
* We Split the Datset into Training Data and Testing Data.

# Step 8: Training the Model.
* We use Logistic Regression() for training the Model.

# Step 9: Model Evalution.
* We evaluate the accuracy of the Model using the 'accuray_score()' function.

# Step 10: Making System Prediction.
* We make prediction if the News is Real or Fake.
