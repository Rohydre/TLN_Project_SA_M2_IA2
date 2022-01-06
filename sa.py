import xml.etree.ElementTree as ET
import os
import nltk
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


# If needed download stopwords
# nltk.download('stopwords')


# Utilities functions to parse xml.

# Input path to XML file and returns ["text", "aspectTerm", "aspectCategory"]
def extract_dataset(path):
    dataset = []
    data = ET.parse(path)
    root = data.getroot()
    for node in root.iter('sentence'):
        question = node[0].text
        for aspectTerm in node.iter('aspectTerm'):
            term = aspectTerm.attrib.get('term')
            polarity = aspectTerm.attrib.get('polarity')
            if polarity != 'conflict':
                dataset.append([question, term, polarity])
    return dataset


# Wrapper function to extract dataset and put it in list
def format_dataset(xml_path):
    xml_path = os.getcwd().replace("\\", "/") + xml_path
    nodes = []
    dataset = extract_dataset(xml_path)
    for node in dataset:
        nodes.append(node)
    return nodes

#Tokenizer
def tok(doc, sw=False):
    result = []
    for line in doc:
        tokenized = nltk.word_tokenize(line)
        if sw:
            stop_words = set(stopwords.words('english'))
            l = [word for word in tokenized if word not in stop_words and word.isalpha()]
            result.append(l)
        else:
            l = [word for word in tokenized]
            result.append(l)
    return result

#Bag of words
def bow(doc, vectorizer=None):
    doc = tok(doc, sw=True)
    porter = PorterStemmer()
    docs = []
    for d in doc:
        li = []
        for word in d:
            li.append(porter.stem(word))
        docs.append(" ".join(li))
    if vectorizer is None:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(docs)
    else:
        bow = vectorizer.transform(docs)
    bow = bow.toarray()
    bow = pd.DataFrame(bow)
    return bow, vectorizer


def sentiment_analysis(file_train, file_test):
    # We will use Random Forest Classifier
    clf = RandomForestClassifier()

    # Get and format training and testing dataset from file
    dataset = format_dataset(file_train)
    dataset_test = format_dataset(file_test)

    # First we train the classifier with the training dataset
    # 1 - Build x and y variables
    x_train = []
    y_train = []
    for line in dataset:
        x_train.append(line[0])
        y_train.append(line[2])

    if(DEBUG):
        print("############# TRAIN ############")
        print(len(y_train))
        print(len(x_train))

   
    # 2 -Vectorize dataset with bag of words:
    data = bow(x_train)
    x_train = data[0]
    vectorizer = data[1]

    if(DEBUG):
        print(x_train)

    # 3 - Train RandomForest on the training dataset
    clf.fit(x_train, y_train)

    # Then we the test the classifier on the test dataset
    # 1 - Build x and y variables
    x_test = []
    y_test = []
    for line in dataset_test:
        x_test.append(line[0])
        y_test.append(line[2])

    if(DEBUG):
        print("############# TEST ############")
        print(len(y_test))
        print(len(x_test))


    # 2 -Vectorize dataset with bag of words and add x_test to previous x_train
    data = bow(x_test, vectorizer)
    x_test = data[0]

    if(DEBUG):
        print(x_test)

    # 3 - Predict with RandomForest on the testing dataset
    y_pred_test = clf.predict(x_test)


    # 4 - Print the result
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("MCC:", matthews_corrcoef(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("F-measure:", f1_score(y_test, y_pred_test, average='weighted'))

    # 5 - Return the prediction (not used)
    return y_pred_test

DEBUG = False

if __name__ == '__main__':
    # On va exécuter notre pipeline sur les différents jeu de données fournis au format xml
    print("###############################################")
    print("RESTAURANTS")
    sentiment_analysis("/xml/Restaurants_Train.xml", "/xml/Restaurants_Test_Gold.xml")
    print("###############################################")

    print("###############################################")
    print("LAPTOP")
    sentiment_analysis("/xml/Laptop_Train.xml", "/xml/Laptop_Test_Gold.xml")
    print("###############################################")
    