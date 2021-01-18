import tweepy
import pandas as pd
import re

import operator
import gc
import string
import time
from time import sleep
from time import time
import datetime
import json
import random
import math
import pandas as pd
import os
from os import listdir

#Parameters to configure:<br>

#*   personsOfInterestList - Array of usernames of twitter users you wish to extract
#*   limitCount - Maximum number of tweets per person
#*   minimumNumberOfWordPerTweet - Do not collect tweets with less than N words in them
#*   shouldCollectComments - Collect user's comments in addition to tweets

def extractTweeterUserInfo(personsOfInterestList,api):

  personsOfInterest_extandedDetails = {}

  informationToGather = ['name','screen_name','description','followers_count','friends_count','profile_image_url','statuses_count','verified','created_at']

  for userName in personsOfInterestList:

    userName = userName.lower()
    personsOfInterest_extandedDetails[userName]={}

    person_tweets = tweepy.Cursor(api.user_timeline,screen_name = userName,tweet_mode="extended",count=2)

    profilePicURL=""
    count=0
    for tweet in person_tweets.items():
      if count==0:
        #print(tweet._json['user'])
        for key,value in tweet._json['user'].items():
          #print(key,value)
          if key in informationToGather:
            if key in 'profile_image_url': #Replace low size image with normal size image
              value = value.replace('normal.jpg','400x400.jpg')
            elif value is None:
              value=""
            personsOfInterest_extandedDetails[userName][key]=value
            #print(userName,key,value)
      count+=1
      if count>0:
        break

  personsOfInterest_extandedDetails = pd.DataFrame(personsOfInterest_extandedDetails).T
  return personsOfInterest_extandedDetails


def getAllTweetsPerPerson(userName, api, limitCount=5000, minimumNumberOfWordPerTweet=4, shouldCollectComments=False):
  twitterCursor = tweepy.Cursor(api.user_timeline, screen_name=userName, tweet_mode="extended", count=limitCount)

  # If tweet starts with an @ = It's a reply to a comment and we do not care about it.
  regexp = re.compile(r'^@')

  person_tweet_dict = {}
  for tweet in twitterCursor.items():

    # Pull the values
    full_text = tweet.full_text
    tweetDate = tweet.created_at
    username = tweet.user.screen_name
    fullName = tweet.user.name
    acctdesc = tweet.user.description
    location = tweet.user.location
    following = tweet.user.friends_count
    followers = tweet.user.followers_count
    totaltweets = tweet.user.statuses_count
    usercreatedts = tweet.user.created_at
    tweetcreatedts = tweet.created_at
    retweetcount = tweet.retweet_count
    hashtags = tweet.entities['hashtags']
    truncated = tweet.truncated
    userMentions = len(tweet.entities['user_mentions'])

    wordCountPerTwit = len(full_text.split())

    try:
      retweeted_status = tweet.retweeted_status
      retweeted = True
    except:
      retweeted_status = None
      retweeted = False

    isComment = False
    if regexp.search(full_text):
      isComment = True

    # If the text is truncated, send an additional call to get full text
    id_str = tweet.id_str

    if not retweeted and wordCountPerTwit > minimumNumberOfWordPerTweet and userMentions <= 2 and (
            not isComment or shouldCollectComments):
      twitDict = {
        "full_text": full_text
        , "tweetDate": tweetDate
        , "username": username
        , "fullName": fullName
        , "acctdesc": acctdesc
        , "location": location
        , "following": following
        , "followers": followers
        , "totaltweets": totaltweets
        , "usercreatedts": usercreatedts
        , "tweetcreatedts": tweetcreatedts
        , "retweetcount": retweetcount
        , "hashtags": hashtags
        , "truncated": truncated
        , "userMentions": userMentions
        , 'wordCountPerTwit': wordCountPerTwit
      }

      person_tweet_dict[str(id_str)] = twitDict
      # print("%s | %s - %s"%(username,tweetDate,full_text))

      if len(person_tweet_dict) == limitCount:
        break

  print("Completed %s, releavnt tweets: %s" % (username, len(person_tweet_dict)))
  return person_tweet_dict


def extractTweetsForListOfUsers(personsOfInterestList, baseFolder_models,
                                baseFolder_static,
                                extractTwitsRegardlessIfExists=True, tweetsPerPerson=1000,
                                minimumNumberOfWordPerTweet=4, shouldCollectComments=False, shouldSaveAsCSV=True):
  # Collect Twitter API keys
  # For security, keys are saved locally on drive as a txt file:

  apiKeyPath = os.path.join(baseFolder_static,'TwitterAPIKeys.txt')

  if not os.path.exists(apiKeyPath):
    print(
      'ApiKey path was NOT found. Please make sure that you have a text file with your consumer_key,consumer_secret and bearer_token save locally')

  consumer_key = pd.read_csv(apiKeyPath, header=None).T[0][0]
  consumer_secret = pd.read_csv(apiKeyPath, header=None).T[1][0]
  bearer_token = pd.read_csv(apiKeyPath, header=None).T[2][0]

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  api = tweepy.API(auth, wait_on_rate_limit=True)

  savePathTwits = os.path.join(baseFolder_models, 'Raw Tweet Data.csv')
  savePathUsersInfo = os.path.join(baseFolder_models, 'Extended User Information.csv')

  if extractTwitsRegardlessIfExists or not os.path.exists(
          savePathTwits) or not os.path.exists(
          savePathUsersInfo):  # If not previously scraped or if requested override - Override
    completeListOfPeopleAndTheirTweets = {}

    for userName in personsOfInterestList:

      userName = userName.lower()
      print('UserName: %s' % userName)
      Finished = False
      while Finished is False:
        try:
          person_tweet_dict = getAllTweetsPerPerson(userName=userName, api=api, limitCount=tweetsPerPerson,
                                                    minimumNumberOfWordPerTweet=minimumNumberOfWordPerTweet,
                                                    shouldCollectComments=shouldCollectComments)
          Finished = True
        except Exception as e:
          print('Operation failed for user: %s. Will try again in 30 seconds' % (userName))
          print("Error: \n%s" % e)
          Finished = False
          sleep(30)  # wait 30 seconds

      _ = completeListOfPeopleAndTheirTweets.update(person_tweet_dict)

    # shuffle tweets:

    tweetIdList = list(completeListOfPeopleAndTheirTweets.items())
    random.shuffle(tweetIdList)
    completeListOfPeopleAndTheirTweets = dict(tweetIdList)

    personsOfInterest_extandedDetails = extractTweeterUserInfo(personsOfInterestList, api=api)

    if shouldSaveAsCSV:
      # Convert dict to Dataframe and use the csv functionality
      completeListOfPeopleAndTheirTweets = pd.DataFrame(completeListOfPeopleAndTheirTweets).T.fillna("")
      completeListOfPeopleAndTheirTweets.to_csv(savePathTwits, encoding='utf-8-sig',index_label='tweetId')

      personsOfInterest_extandedDetails.to_csv(savePathUsersInfo, encoding='utf-8-sig')

  else:

    print("Will load tweet data from existing files: %s"%(savePathTwits))
    print("Will load udser data from existing files: %s" % (savePathUsersInfo))
    completeListOfPeopleAndTheirTweets = pd.read_csv(savePathTwits,index_col='tweetId')
    personsOfInterest_extandedDetails = pd.read_csv(savePathUsersInfo,header=0,index_col=0).fillna("")

  return completeListOfPeopleAndTheirTweets, personsOfInterest_extandedDetails

#test_personsOfInterestList=['Rihanna','BarackObama']
#test_baseFolder = 'models/testModel'
#test_projectName = 'testProject'
#test_completeListOfPeopleAndTheirTweets,test_personsOfInterest_extandedDetails = extractTweetsForListOfUsers(projectName=test_projectName,personsOfInterestList=test_personsOfInterestList,baseFolder=test_baseFolder,extractTwitsRegardlessIfExists=True,tweetsPerPerson=1000,minimumNumberOfWordPerTweet=4,shouldCollectComments=False,shouldSaveAsCSV=True)
#
#print(test_completeListOfPeopleAndTheirTweets)