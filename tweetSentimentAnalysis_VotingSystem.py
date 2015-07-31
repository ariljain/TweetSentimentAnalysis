"""
Predict the sentiment of a tweet captured in real time
Currently tweet is classification into positive or negative
With little enhancments a range of emotions can be handled 

External Modules Used :
tweepy       : acessing twitter streaming API
nltk         : processing text data
scikit-learn : supervised learning classifiers

Method   : Voting system using multiple classifiers

Classifier Models Used: 
Multinomial Bayes, Linear Support Vector Classification, 
Logistic Regression, Stochastic Gradient Descent
Nu-Support Vector Classification
"""

import sys
import time
import random
import json
import re
from statistics import mode

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import nltk
from nltk.classify import ClassifierI

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC


#list of keywords that needs to be used to filter tweets
#Can add as many keywords as needed in the list
#If the list gets too long, you might start missing some tweets 
#due to extra filtering time needed
trackingKeywords = ['dummyKeyword1','dummyKeyword2']

#Special tags that can be generally found in a tweet
USERNAME = re.compile('@[A-Za-z0-9]+',re.IGNORECASE)
HASHTAG = re.compile('#[A-Za-z0-9]+',re.IGNORECASE)
HASHSIGN = re.compile('#')
URL = re.compile('(((https?|ftp|file)(:)(\/\/)?)|(www.))[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]',re.IGNORECASE)
RETWEET = re.compile('^(rt)|^(RT)')
NON_APLHA = re.compile('[^a-zA-Z]')
MULTI_SPACE = re.compile('\s{2,10}')
START_SPACE = re.compile('^\s{1,10}')


#function to remove URL, hasgtag and usernames
def removeTags(text):
    """
    This function finds URLs, Hashtags, Username, multiple spaces in the text passed
    Replaces them with empty string
    """
    try:
        text = re.sub(USERNAME,'',text)
        text = re.sub(HASHTAG,'',text)
        text = re.sub(URL,'',text)
        text = re.sub(RETWEET,'',text,1)
        text = re.sub(MULTI_SPACE,'',text,1)
        text = re.sub(START_SPACE,'',text,1)
        return text

    except BaseException as e:
        print("\n***Inside removeTags()")
        print("***ERROR:",str(e))


#Vectorizer for pre-processing text
count_vect = CountVectorizer(stop_words="english",strip_accents='unicode',ngram_range=(1, 3))

#function to extract features of the new data
def extractFeatures(data):
    """
    This function extracts features from the list of given data
    data must be a list of tweets
    """
    try:
        global count_vect
        X_new_counts = count_vect.transform(data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        #print("Done extracting features !")
        return X_new_tfidf

    except BaseException as e:
        print("\n***Inside extractFeaures()")
        print("***ERROR:",str(e))


#class to handle mutiple classifier predictions and calculate confidence
class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        """
        for every classifier passed
            predict the sentiment
            save the sentiment in vote[] list
            return the sentiment with highest votes
        """
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])

        return mode(votes)

    def confidence(self, features):
        """
        for every classifier passed
            predict the sentiment
            save sentiment in vote[] list
            calculate confidence : ratio of max votes to a particular sentiment / total votes
        """
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


#File containing initial training data
#Training Data Set : http://pythonprogramming.net/static/downloads/short_reviews/
short_pos = open("FILE PATH CONTAINING POSITIVE SENTIMENT DATA","r")
short_neg = open("FILE PATH CONTAINING NEGATIVE SENTIMENT DATA","r")
#short_neu = open("neutral_tweets.txt","r")

#list that will hold all the training data in the form
#[("tweet_text1","sentiment1"),("tweet_textN","sentimentN"),...,("tweet_textN","sentimentN")]
documents = []

print("Starting to load training data")

for line in short_pos:
    documents.append( (line, "pos") )

for line in short_neg:
    documents.append( (line, "neg") )

"""
for line in short_neu:
    documents.append( (line, "neu") )
"""

#randomizing the order of training data for better training
random.shuffle(documents)

#list of possible sentiments
sentiments = ['pos','neg','neu']

#list of all training text
reviews = []

#list of sentiment lables of 
review_sentiments = []

for x in documents:
    reviews.append(x[0])

    if x[1] == 'pos':
        review_sentiments.append(0)
    elif x[1] == 'neg':
        review_sentiments.append(1)
    else:
        review_sentiments.append(2)

print("Finished loading training data")
print("Starting to pre-process training data")

#pre-processing training data : tokenizing and filtering of stopwords
try:
    X_train_counts = count_vect.fit_transform(reviews)
    print("Finished tokenizing and filtering of stopwords...")

except BaseException as e:
    print("\n***While pre-processing using CountVectorizer")
    print("***ERROR:",str(e))


"""
Calculating : tf and tf–idf 

Issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.

To avoid these potential discrepancies
    divide the number of occurrences of each word in a document by the total number of words in the document
    these new features are called tf for Term Frequencies.

Refinement: downscale weights for words that occur in many documents in the corpus
            as they are less informative than those that occur only in a smaller portion of the corpus
            This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
"""
try:
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print("X_train_tfidf.shape",X_train_tfidf.shape)
    print("Finished calculating tf and tf-idf...")

except BaseException as e:
    print("\n***While calculating tf and tf-idf")
    print("***ERROR:",str(e))



#training classifiers

startTime = time.time()
print("\nStarting to train MultinomialNB...")

try:
    #Multinomial Bayesian Calssifier
    clf_MultiNB = MultinomialNB().fit(X_train_tfidf, review_sentiments)
    print("Finished training MultinomialNB in %.4f seconds" % float(time.time()-startTime))

except BaseException as e:
    print("\n***While training Multinomial Bayes Classifier")
    print("***ERROR:",str(e))


startTime = time.time()
print("\nStarting to train SVC...")

try:
    #Linear Support Vector Classifier
    clf_SVC = LinearSVC(loss='l2', penalty=penalty, 
                        dual=False, tol=1e-3).fit(X_train_tfidf, review_sentiments)
    print("Finished training Linear SVC in %.4f seconds" % float(time.time()-startTime))

except BaseException as e:
    print("\n***While training Linear Support Vector Classifier")
    print("***ERROR:",str(e))


startTime = time.time()
print("\nStarting to train NuSVC...")

try:
    #NuSVC Calssifier
    clf_NuSVC = NuSVC(nu=0.5, kernel='rbf', degree=3,
                      gamma=0.0, coef0=0.0).fit(X_train_tfidf, review_sentiments)
    print("Finished training NuSVC in %.4f seconds" % float(time.time()-startTime))

except BaseException as e:
    print("\n***While training Nu-Support Vector Classifier")
    print("***ERROR:",str(e))


startTime = time.time()
print("\nStarting to train LogisticRegression...")

try:
    #Logistic Regression Classifier
    clf_LogRegress = LogisticRegression().fit(X_train_tfidf, review_sentiments)
    print("Finished training LogisticRegression in %.4f seconds" % float(time.time()-startTime))

except BaseException as e:
    print("\n***While training Logistic Regression Classifier")
    print("***ERROR:",str(e))


startTime = time.time()
print("\nStarting to train SGDC...")

try:
    #Stochastic Gradient Descent Classifier
    clf_SGDC = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, 
                             l1_ratio=0.15, fit_intercept=True, n_iter=5,
                             shuffle=True).fit(X_train_tfidf, review_sentiments)
    print("Finished training SGDC in %.4f seconds" % float(time.time()-startTime))

except BaseException as e:
    print("\n***While training SGD Classifier")
    print("***ERROR:",str(e))

print("\nInitializing voting setup")
voted_classifier = VoteClassifier(
                                  clf_MultiNB, clf_SVC,
                                  clf_NuSVC, clf_LogRegress,
                                  clf_SGDC)

#function to find sentiment of tweet
def sentiment(tweet):
    try:
        global voted_classifier
        features = extractFeatures(tweet)
        return voted_classifier.classify(features),voted_classifier.confidence(features)
    
    except BaseException as e:
        print("\n***Inside sentiment()")
        print("***ERROR:",str(e))

#consumer key, consumer secret, access token, access secret.
CONSUMER_KEY   ="PUT YOUR CONSUMER_KEY HERE"
CONSUMER_SECRET="PUT YOUR CONSUMER_SECRET HERE"
ACCESS_TOKEN   ="PUT YOUR ACCESS_TOKEN HERE"
ACCESS_SECRET  ="PUT YOUR ACCESS SECRET HERE"

print("\nStarting Twitter streaming...")

class listener(StreamListener):

    def on_data(self, data):
        try:
            #capture tweet in json format
            all_data = json.loads(data)

            #remove username, hashtags and 
            tweet = []
            tweet.append(removeTags(all_data["text"]))

            try:
                #getting sentiment and confidence value for the tweet
                sentiment_value, confidence = sentiment(tweet)
                print(all_data["text"].encode('utf-8'), sentiments[sentiment_value], confidence)
            
            except BaseException as e:
                print("\n***Inside on_data : trying to get sentiment")
                print("***ERROR:",str(e))

            if confidence*100 >= 80:
                #if more than 3 classifiers agree on a sentiment
                output = open("OUTPUT FILE NAME","a")
                output.write(sentiments[sentiment_value])
                output.write('\n')
                output.close()

            return True            
        
        except BaseException as e:
            print("\n***Inside on_data")
            print("***ERROR:",str(e))
            return True #Don't kill the stream 

    def on_error(self, status):
        print('Error Status:',status)
        return True # Don't kill the stream

    def on_timeout(self):
        print ('Timeout...\n')
        return True # Don't kill the stream

print("Setting up OAuth for Twitter Streaming API...")

auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

print("Setting up streaming listener...")

twitterStream = Stream(auth, listener())

print("Starting to capture twitter streams...")

twitterStream.filter(track=trackingKeywords,languages=['en'])