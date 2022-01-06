# M2 IA2 NLT PROJECT - SOUMRANY MAXIME
# Sentiment Analysis

A school project about natural language processing during my Master Degree for sentiment analysis

Given the dataset in the xml folder, the program was able to give these results:

```
###############################################
RESTAURANTS
Accuracy: 0.90625
MCC: 0.7965903218826272
Precision: 0.9096109525899913
Recall: 0.90625
F-measure: 0.9070526695526696
###############################################
###############################################
LAPTOP
Accuracy: 0.9591836734693877
MCC: 0.9243169000478679
Precision: 0.9618169848584595
Recall: 0.9591836734693877
F-measure: 0.9580770557070755
###############################################
```

Those result are pretty good given the state of art from other systems:
```
Name        Accuracy 
NRC-Can.    82.92
XRCE        78.14
UNITOR      76.29
```

## Requirements

- Python 3.6+
- NLTK and NLTK corpus and stopwords downloaded.
- sklearn installed
- pandas installed

If needed install the required module using `pip install nltk` `pip install sklearn` `pip install panda`
And download the stopwords from nltk `nltk.download('stopwords')` (Uncomment the line in sa.py)
 
## How it works

**Preprocess**
1. Tokenize the words
2. Remove stopwords
3. Use Porter Stemmer to get word's root
4. Apply a Bag Of Word algoritm to get Sentences as Row and Words as Collumns

**Process**
1. Run the preprocess to extract the dataset
2. Initialize Random Forest with initial settings.
3. Train Random Forest Model providing it the training dataset and results.
4. Predict using the trained model on the test dataset

**Results**
1. We then compare our prediction with the gold result given in the data on different metrics:
    - Accuracy
    - MCC
    - Precision
    - Recall
    - F-measure

The program also return its prediction in case it could be used elsewhere

## Starting the program

To start the program, in the command prompt, run the following command:
`python sa.py` (or `python3 sa.py` if you have different python version installed)

If you renamed the source file, rename sa.py accordingly 

## Limits

The program is limited in different aspect:

- We are strongly tied to the xml formatting provided.