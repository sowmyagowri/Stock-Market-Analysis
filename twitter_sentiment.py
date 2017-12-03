import os
import sys
import tweepy
import requests
import numpy as np
from textblob import TextBlob

consumer_key = 'ICKsvPpVq8O4YWSzFNyQRQGFw'
consumer_secret = 'L0MlmOCfOThlHwe19FRJsfQ5iYB5gZ6Qh3ASV9f0SBoX9O8rDV'
access_token = '900139361915641856-v1aedUJsF9UCdmO0Z9KQkeSJDfBq9fJ'
access_token_secret = 'ad0cOH8AzhJtLKyuqUJZOybSdsvianRpJyUwpyK4nQX6U'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

def stock_sentiment(quote, num_tweets):
    now=datetime.datetime.now().date()
    y = now - timedelta(1)
    dby = y - timedelta(1)

    list_of_tweets = user.search(quote, count=num_tweets, since=str(y), until=str(now))
    polarity_list = []
    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        polarity = blob.polarity
        polarity.append(polarity)

    return (sum(polarity_list)/num_tweets)

stock_sentiment('TWTR', 500)
