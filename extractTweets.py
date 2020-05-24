Extracting tweets from Twitter API

-*- coding: utf-8 -*-

"""

Created on Mon Jan 22 10:56:30 2018 @author: bhama #!/usr/bin/python

import json 
import tweepy 
import csv 
import unicodecsv 
import pandas as pd


#Place your respective keys here 

consumer_key="********************" 
consumer_secret="*************************" 
api_key="***************************" 
api_secret="***********************************” 
auth=tweepy.OAuthHandler(consumer_key,consumer_secret) auth.set_access_token(api_key,api_secret)

class Listeners(tweepy.StreamListener): 
  def on_data(self,data):
    c = unicodecsv.writer(open("C:\\Users\\bhama\\Desktop\\bhama\\major project\\twitter3.csv", "a"))

    streamed=json.loads(data)
    c.writerow([streamed["text"].encode('ascii','ignore'),streamed["user"]["friends_count"],streamed["user"]["followers_count"],streamed["entities"]["user_mentions"],streamed["entities"]["hashtags"],streamed["entities"]["urls"],streamed["user"]["statuses_count"]])
    # c.writerow(rows)

f = open('C:\\Users\\bhama\\Desktop\\bhama\\major project\\keywords.txt', 'r')
array=[]
tweets=tweepy.streaming.Stream(auth,Listeners())

for i in f:
  array.append(i.replace(" ","%100"))

tweets.filter(track=array,async=True)
with open("C:\\Users\\bhama\\Desktop\\bhama\\major project\\twitter3.csv", "wb") as outcsv:

  writer = csv.DictWriter(outcsv, fieldnames = ["text", "friends_count", "followers_count","user_mentions","hashtags","urls","statuses_count"])

writer.writeheader()
