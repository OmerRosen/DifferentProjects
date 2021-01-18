from omerprojects.TwitterClassifierFolder.pythonFiles.Scraper import extractTweetsForListOfUsers
from omerprojects.TwitterClassifierFolder.pythonFiles.DataPreProcessing import tweet_to_BOW
from omerprojects.TwitterClassifierFolder.pythonFiles.NLP_LDA_Class import LDA_Perdictions
from gensim.models import LdaModel
import os
import pandas as pd
import pickle
import numpy as np
import json
from difflib import SequenceMatcher

def datePreProcessingForUser(userExtractedTweets, traininDatasetHeaders):
    exampleDatasetDict = {header: 0 for header in traininDatasetHeaders}

    randomUserAndTheirTweets_BOW = {'testData': exampleDatasetDict}
    randomUserOriginalTweets = []

    m=0

    if type(userExtractedTweets)==dict:
        userExtractedTweets = pd.DataFrame(userExtractedTweets).T
        userExtractedTweets['id_str'] = userExtractedTweets.index
        columnsOrder = ['id_str','full_text','tweetDate','username','fullName','acctdesc','location','following','followers','totaltweets','usercreatedts','tweetcreatedts','retweetcount','hashtags','truncated','userMentions']
        userExtractedTweets = userExtractedTweets.filter(items=columnsOrder)

    username = None
    for item in userExtractedTweets.head(100000).values:

        if username != item[3]:
            print('Finished pre-processing %s, moving to handle: %s' % (username, item[3]))

        id_str = str(item[m+0])
        full_text = item[m+1]
        tweetDate = item[m+2]
        username = item[m+3]
        fullName = item[m+4]
        acctdesc = item[m+5]
        location = item[m+6]
        following = item[m+7]
        followers = item[m+8]
        totaltweets = item[m+9]
        usercreatedts = item[m+10]
        tweetcreatedts = item[m+11]
        retweetcount = item[m+12]
        hashtags = item[m+13]
        truncated = int(item[m+14])
        userMentions = item[m+15]

        wordCountPerTwit = len(full_text.split())

        BOWDict = tweet_to_BOW(full_text, printTxt=False)
        BOWDict['aa_UserName'] = username
        BOWDict['aa_truncated'] = truncated
        BOWDict['aa_userMentions'] = userMentions
        BOWDict['aa_wordCountPerTwit'] = wordCountPerTwit

        features = list(BOWDict.keys())
        for feature in features:
            if feature not in traininDatasetHeaders:
                del BOWDict[feature]

        randomUserOriginalTweets.append(full_text)

        randomUserAndTheirTweets_BOW[id_str] = BOWDict

    randomUserAndTheirTweets_df = pd.DataFrame(randomUserAndTheirTweets_BOW)
    randomUserAndTheirTweets_df = randomUserAndTheirTweets_df.T.fillna(0)
    randomUserAndTheirTweets_df = randomUserAndTheirTweets_df.filter(items=traininDatasetHeaders)
    randomUserAndTheirTweets_df = randomUserAndTheirTweets_df.drop('testData')

    tweetsAndTheirTxt_Pd = pd.DataFrame(randomUserOriginalTweets, index=randomUserAndTheirTweets_df.index,
                                        columns=["full_tweet"])

    return randomUserAndTheirTweets_df, tweetsAndTheirTxt_Pd


def classifyUserTweets_BasedOnModel(randomUsername, winningModel, winningLDAModel, dataSet, tweetsAndTheirTxt_Pd,personsOfInterestDict):
    predict_proba = winningModel.predict_proba(dataSet)

    probabilityThreshold = 0.3

    personsOfInterestDict_Reversed = {val:key for key,val in personsOfInterestDict.items()}

    twitsAndTheirSimilarity = {}
    selectedUserStatistics = {author: 0 for author in personsOfInterestDict_Reversed.keys()}
    selectedUserStatistics['unassigned'] = 0

    allTweets = tweetsAndTheirTxt_Pd.values
    tstTweet = np.array2string(allTweets[0])
    #print(tstTweet)
    #print(len(allTweets))

    for i, pred in enumerate(predict_proba):
        twit_Id = dataSet.index[i]
        original_twit = allTweets[i][0]
        maxPredIdx = np.argmax(pred)
        prodPercent = pred[maxPredIdx]
        similarAuthor = personsOfInterestDict[str(maxPredIdx)]
        htmlFormat = winningLDAModel.convert_string_into_HTML(original_twit, wordsAndTheirTopics=None,fontSize_px=26)

        for idx, prob in enumerate(list(pred)):
            if prob >= probabilityThreshold:
                author = personsOfInterestDict[str(idx)].lower()
                selectedUserStatistics[author] += 1

        if not prodPercent >= probabilityThreshold:
            selectedUserStatistics['unassigned'] += 1

        if i % 50 == 0:
            # print(i,pred)
            # print(maxPredIdx,prodPercent,similarAuthor)
            pass
        twitSimilarityDict = {
            'original_Author': randomUsername,
            'original_twit': original_twit,
            'similarAuthor': similarAuthor,
            'probPercent': prodPercent,
            'html_Format': htmlFormat
        }
        twitsAndTheirSimilarity[twit_Id] = twitSimilarityDict

    twitsAndTheirSimilarity = pd.DataFrame(twitsAndTheirSimilarity).T

    #print(selectedUserStatistics)
    print("\n")

    return twitsAndTheirSimilarity, selectedUserStatistics



def analyseTweetsForRandomUser(randomUsername,extandedProjectName,shouldCollectComments=True,fontSize_px=22,nTweetsToCompareFromAuthor = 300,baseFolder=""):



    if baseFolder == "":
        #baseFolder = os.getcwd()
        baseFolder=r"C:\Users\omerro\Google Drive\Data Science Projects\OmersPortal\omerprojects\TwitterClassifierFolder"

    baseFolder_static = os.path.join(baseFolder, 'static')
    baseFolder_models = os.path.join(baseFolder, 'models', extandedProjectName)
    baseFolder_Hyperparameters = os.path.join(baseFolder_models,'Model_HyperParameters.txt')
    baseFolder_RawTweetData = os.path.join(baseFolder_models,'Raw Tweet Data.csv')
    baseFolder_classifierPath = os.path.join(baseFolder, 'models', extandedProjectName,"Classifier\Classifier.pkl")
    baseFolder_trainingData = os.path.join(baseFolder, 'models', extandedProjectName,"Classifier\TrainingData_X_Y_YNames.pkl")
    baseFolder_LDA = os.path.join(baseFolder_models, 'LDA')
    baseFolder_Classifier = os.path.join(baseFolder_models, 'Classifier')

    tweetsPerPerson = 3000

    if not os.path.exists(baseFolder_models):
        errorMSG = "Model path does not exists: %s"%(baseFolder_models)
        print(errorMSG)
        raise ValueError(errorMSG)
    else:

        with open(baseFolder_Hyperparameters) as json_file:
            modelParameterJson = json.load(json_file)

        personsOfInterestList=modelParameterJson['personsOfInterestList']
        personsOfInterestDict=modelParameterJson['personsOfInterestDict']
        personsOfInterestDict_Reversed=modelParameterJson['personsOfInterestDict_Reversed']
        personsOfInterest_extandedDetails=modelParameterJson['outputs']['personsOfInterest_extandedDetails']


        completeListOfPeopleAndTheirTweets, randomUser_extandedDetails = extractTweetsForListOfUsers(personsOfInterestList=randomUsername, baseFolder_models=baseFolder_models,
                                                                                                     baseFolder_static=baseFolder_static,
                                                                                                     extractTwitsRegardlessIfExists=True, tweetsPerPerson=tweetsPerPerson,
                                                                                                     minimumNumberOfWordPerTweet=4, shouldCollectComments=shouldCollectComments, shouldSaveAsCSV=False)

        #print(personsOfInterest_extandedDetails)


        selectedClassifier = instanceOfThisDumbClass_FromPickle = pickle.load(open(baseFolder_classifierPath, "rb"))
        X,Y,YNames = instanceOfThisDumbClass_FromPickle = pickle.load(open(baseFolder_trainingData, "rb"))
        traininDatasetHeaders = list(X.columns.values)


        randomUserAndTheirTweets_df, tweetsAndTheirTxt_Pd = datePreProcessingForUser(userExtractedTweets=completeListOfPeopleAndTheirTweets, traininDatasetHeaders=traininDatasetHeaders)

        #ldaPredictionModel = LdaModel.load(baseFolder_LDA)
        ldaPredictionModel = LDA_Perdictions(baseFolder_LDA)
        ldaPredictionModel.load_model_Pikcle()

        randomUserAndTheirTweets_df_withTopics, new_Corpus = ldaPredictionModel.predict_On_Unseen_Corpus(tweetsAndTheirTxt_Pd,randomUserAndTheirTweets_df)

        print("Original dataset shape: %s"%(str(X.shape)))
        print("New dataset shape: %s"%(str(randomUserAndTheirTweets_df_withTopics.shape)))

        def Diff(li1, li2):
            return (list(list(set(li1) - set(li2)) + list(set(li2) - set(li1))))

        differenceBetweenDatasets = Diff(list(X.columns), list(randomUserAndTheirTweets_df_withTopics.columns))
        if differenceBetweenDatasets!=[]:

            print("Datasets are Not in the same shape.\d Difference:")
            print("Datasets are Not in the same shape")
            print(differenceBetweenDatasets)

        listOfTweets = list(completeListOfPeopleAndTheirTweets.keys())
        listOfTexts = [val['full_text'] for val in completeListOfPeopleAndTheirTweets.values()]
        tweetsAndTheirTxt_Pd=pd.DataFrame(listOfTexts)
        tweetsAndTheirTxt_Pd.index=listOfTweets

        twitsAndTheirSimilarity, selectedUserStatistics = classifyUserTweets_BasedOnModel(randomUsername=randomUsername,
                                                                                          winningModel=selectedClassifier,
                                                                                          winningLDAModel=ldaPredictionModel,
                                                                                          dataSet=randomUserAndTheirTweets_df_withTopics,
                                                                                          tweetsAndTheirTxt_Pd=tweetsAndTheirTxt_Pd,
                                                                                          personsOfInterestDict=personsOfInterestDict)



        originalRawTweetData = pd.read_csv(baseFolder_RawTweetData,index_col="tweetId")

        dfUsersAndTheirTweets = originalRawTweetData.filter(items=['tweetId','full_text']).copy()
        dfUsersAndTheirTweets['aa_UserName'] = originalRawTweetData['username'].str.lower()

        dictOfTweetSimilarties = {}

        count = 0

        alreadyUsedTweets = []

        for i, tweetId in enumerate(twitsAndTheirSimilarity.index):
            dictOfTweetSimilarties[tweetId] = {}
            original_Author = twitsAndTheirSimilarity.loc[tweetId, 'original_Author']
            original_twit = twitsAndTheirSimilarity.loc[tweetId, 'original_twit']
            originalTwitHTMLFormat = ldaPredictionModel.convert_string_into_HTML(original_twit, wordsAndTheirTopics=None, fontSize_px=fontSize_px)
            similarAuthor = twitsAndTheirSimilarity.loc[tweetId, 'similarAuthor']
            probPercent = twitsAndTheirSimilarity.loc[tweetId, 'probPercent']
            if i == 1:
                print(original_Author, similarAuthor, probPercent, original_twit)

            topNTweetsFromAuthor = dfUsersAndTheirTweets[dfUsersAndTheirTweets['aa_UserName'] == similarAuthor].head(nTweetsToCompareFromAuthor).values
            topNTweetIdsFromAuthor = dfUsersAndTheirTweets[dfUsersAndTheirTweets['aa_UserName'] == similarAuthor].head(nTweetsToCompareFromAuthor).index
            biggestSimilarity = 0
            for j, item in enumerate(topNTweetsFromAuthor):
                authorTweet = item[1]

                authorTweetId = topNTweetIdsFromAuthor[j]

                similarity = SequenceMatcher(None, original_twit, authorTweet).ratio()
                if similarity > biggestSimilarity and authorTweetId not in alreadyUsedTweets:
                    biggestSimilarity = similarity
                    authorTweetHTMLFormat = ldaPredictionModel.convert_string_into_HTML(authorTweet, wordsAndTheirTopics=None, fontSize_px=fontSize_px)

                    authorTweetId_max = authorTweetId

                    dictOfTweetSimilarties[tweetId] = {
                        'original_Author': original_Author,
                        'similarAuthor': similarAuthor,
                        'probPercent': probPercent,
                        'biggestSimilarity': biggestSimilarity,
                        'original_twit': original_twit,
                        'authorTweet': authorTweet,
                        'originalTwitHTMLFormat': originalTwitHTMLFormat,
                        'authorTweetHTMLFormat': authorTweetHTMLFormat,
                        'authorTweetId': authorTweetId

                    }
                if j == 0 and i == 0:
                    print(original_twit)
                    print(authorTweet)
                    print(similarity)

            alreadyUsedTweets.append(authorTweetId_max)

            count += 1

            if count > 200:
                break

        distanceBetweenTweetsHTMLFormat = pd.DataFrame(dictOfTweetSimilarties).T.sort_values(by=['biggestSimilarity'],
                                                                                             ascending=False).head(10)

    twitsAndTheirSimilarity = twitsAndTheirSimilarity.sort_values(by=['probPercent'], ascending=False)
    twitsAndTheirSimilarity['probPercent'] = twitsAndTheirSimilarity['probPercent'].astype(float)

    twitsAndTheirSimilarity_summary = twitsAndTheirSimilarity.groupby(['similarAuthor']).agg(TotalTweets=('original_twit', 'size'),
                                                           Probability_Percent=('probPercent', 'mean'),
                                                           Probability_Max=('probPercent', 'max'),
                                                           Probability_Min=('probPercent', 'min')
                                                           )

    twitsAndTheirSimilarity_summary = twitsAndTheirSimilarity_summary.sort_values('Probability_Percent',ascending=False)
    twitsAndTheirSimilarity_summary.insert(0,'matchRank',range(1,1+len(twitsAndTheirSimilarity_summary)))

    #print(twitsAndTheirSimilarity_summary,randomUser_extandedDetails,personsOfInterest_extandedDetails)
    return  twitsAndTheirSimilarity_summary.sort_values('Probability_Percent',ascending=False),randomUser_extandedDetails,personsOfInterest_extandedDetails

if __name__=="__main__":




    randomUsername=['KKajderowicz']
    extandedProjectName = 'FinalProject_5Classes_20twtpp_authors_Jo_la_ia_El_St'

    shouldCollectComments = True
    fontSize_px = 22
    nTweetsToCompareFromAuthor = 300

    twitsAndTheirSimilarity_summary,randomUser_extandedDetails,personsOfInterest_extandedDetails = analyseTweetsForRandomUser(randomUsername=randomUsername, extandedProjectName=extandedProjectName, shouldCollectComments=shouldCollectComments, fontSize_px=fontSize_px,
                               nTweetsToCompareFromAuthor=nTweetsToCompareFromAuthor,baseFolder="")

    print ("twitsAndTheirSimilarity_summary")
    print (twitsAndTheirSimilarity_summary)
    print("randomUser_extandedDetails")
    print(randomUser_extandedDetails)
    print("personsOfInterest_extandedDetails")
    print(personsOfInterest_extandedDetails)

