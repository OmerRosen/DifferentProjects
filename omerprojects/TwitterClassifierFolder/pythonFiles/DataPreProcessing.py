from urllib.parse import urlparse
import string
import time
from time import sleep
import json
import pandas as pd
import os
from time import time
import emoji
from profanity_check import predict_prob as profanity_prob

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import progressbar

import requests
from bs4 import BeautifulSoup
import pickle

def saveLoadStopwords(baseFolder='',shouldLoadPretrainedStopwordList=True,updatedStopwordList=[]):

    if baseFolder=="":
        baseFolder= os.path.join(os.getcwd(),"omerprojects/TwitterClassifierFolder/static")

    loadPretrainedStopwordList_Path = os.path.join(baseFolder, 'UpdatedStopWords.csv')
    if os.path.exists(loadPretrainedStopwordList_Path) and shouldLoadPretrainedStopwordList and updatedStopwordList==[]:
      updatedStopWords = pickle.load(open(loadPretrainedStopwordList_Path, "rb"))
      updatedStopWords = [word.replace('bow_','') for word in updatedStopWords]
      print("Loaded pre-existing stop-word list - Count: %s"%len(updatedStopWords))
      print(updatedStopWords[50:70])

    elif updatedStopwordList==[]:
      updatedStopWords = list(set(stopwords.words("english")))
    else: #Save the file
        pickle.dump(updatedStopwordList, open(loadPretrainedStopwordList_Path, "wb"))
        updatedStopWords = updatedStopwordList

    return updatedStopWords

def vocabularyWordList(baseFolder,oveerideCurrentVoc = False,stopWordsList=[]):

    categoriesToExclude = ['adverbs', 'dolch', 'pronouns', 'httpmembersenchantedlearningcomindex', 'verbs',
                           'nounandverb', 'regularverbs', 'adjectives', 'adjectivesforpeople', 'aprilfool']

    path_VocabularyWord = os.path.join(baseFolder,"Vocabulary_Word_Lists.json")
    path_VocabularyCategories = os.path.join(baseFolder,"Vocabulary_Categories_Lists.json")

    #If values exists AND overrid is set to no, load the file:
    if (os.path.exists(path_VocabularyWord) and os.path.exists(path_VocabularyCategories) and oveerideCurrentVoc==False):

        with open(path_VocabularyWord) as json_file:
            vocabularyWord_Dict = json.load(json_file)
        with open(path_VocabularyCategories) as json_file:
            categoryWord_Dict = json.load(json_file)


    # If not - extract all values from website:
    else:

        vocabularyWord_Dict = {}
        categoryWord_Dict = {}

        with open(path_VocabularyWord, 'w') as fp:
            json.dump(vocabularyWord_Dict, fp)
        with open(path_VocabularyCategories, 'w') as fp:
            json.dump(categoryWord_Dict, fp)

        punctuationDict = {x: 0 for x in string.punctuation}
        punctuationDict["’"] = 0

        linkurl = "https://www.enchantedlearning.com/wordlist/"

        page = requests.get(linkurl)

        soup = BeautifulSoup(page.content, 'html.parser')

        results = soup.find_all('a', href=True)

        wordCategoryList = []

        for result in results:
            href = result['href']
            if '/wordlist' in href:
                urlExtantion = linkurl + href.replace('/wordlist/', '')
                wordCategory = href.replace('/wordlist/', '').replace('.shtml', '')
                wordCategory_clean = "".join(
                    [char.lower() for char in wordCategory if char not in punctuationDict.keys()]).strip().replace(' ',
                                                                                                                   '_')

                print(urlExtantion)

                if wordCategory_clean in wordCategoryList:
                    print("Category %s was already covered. Skip duplication" % (wordCategory_clean))
                elif wordCategory in categoriesToExclude:
                    print("Category %s is on the excluded categories list. Will skip.")
                else:

                    categoryWord_Dict[wordCategory_clean] = []

                    internalPage = requests.get(urlExtantion)
                    internalSoup = BeautifulSoup(internalPage.content, 'html.parser')

                    internalResults = internalSoup.find_all(attrs={'class': 'wordlist-item'})

                    for intRez in internalResults:
                        originalWord = intRez.text
                        cleanedWord = "".join([char.lower() for char in originalWord if
                                               char not in punctuationDict.keys()]).strip().replace(' ', '_')

                        if cleanedWord not in stopWordsList and originalWord not in stopWordsList:
                            # print(originalWord,cleanedWord)
                            vocabularyWord_Dict.setdefault(cleanedWord, [])
                            vocabularyWord_Dict[cleanedWord].append(wordCategory_clean)
                            categoryWord_Dict[wordCategory_clean].append(cleanedWord)

                            wordCategoryList.append(wordCategory_clean)

        with open(path_VocabularyWord, 'w') as fp:
            json.dump(vocabularyWord_Dict, fp)
        with open(path_VocabularyCategories, 'w') as fp:
            json.dump(categoryWord_Dict, fp)

    return vocabularyWord_Dict,categoryWord_Dict


def tweet_to_BOW(tweetSample, printTxt=False, measureTime=False, updatedStopWords=[],shouldCollectVocabularyWord=True,vocabularyWord_Dict={}):
    # Start the clock:
    start_time = time()
    new_time = time()

    twitOutputDict = {}

    totalWordCount = 0
    punctuationCount = 0
    upperCaseRatio = 0
    avarageLength = 0
    longestWordLength = 0
    nonExistingWords = 0
    isCurseWord = 0
    hasLink = 0
    LinkBase = 0
    numOfHashtags = 0
    emojiCount = 0

    try:
        isCurseWordProbability = profanity_prob([tweetSample])[0]
    except Exception as e:
        print("Failed on %s" % (tweetSample))
        print(e)
        isCurseWordProbability = 0

    punctuationDict = {}
    punctuationDict["’"] = 0

    totalWordLength = 0
    totalUpperCase = 0
    wordLengthList = [0]

    wordList = tweetSample.split()
    totalWordCount = len(wordList)

    lemmatizer = WordNetLemmatizer()

    if printTxt: print(tweetSample)

    if measureTime:
        timeFromStart = time() - start_time
        timeFromLast = time() - new_time
        new_time = time()
        print('Time from Start: %s, time from last: %s WordNetLemmatizer' % (
        round(timeFromStart), round(timeFromLast, 5)))

    ####################
    # Vocabulary Mapping
    ####################
    def vocabularyWordCatogy(string):
        # print("def vocabularyWordCatogy: %s"%string)
        catg_dict = {}
        if string in vocabularyWord_Dict.keys():
            wordCatgs = vocabularyWord_Dict[string]
            # print("Word %s is in categories: %s"%(string,str(wordCatgs)))
            for catg in wordCatgs:
                dictIdx = "catg_%s" % (catg)
                catg_dict.setdefault(dictIdx, 0)
                catg_dict[dictIdx] += 1

        return catg_dict

    cleanedWordList = []
    ###############################
    # Go over words in the sentence
    ###############################
    for i, word in enumerate(wordList):

        isStopWord = False

        if word in updatedStopWords:
            isStopWord = True

        # print(i,word)

        # Links in twit
        if ('https:' in word or 'http:' in word or 'www.' in word):
            hasLink = 1
            domain = urlparse(word).netloc
            twitOutputDict['url_%s' % domain] = 1
            # Link is NOt counted as a word
            totalWordCount += -1
            if printTxt: print("Word %s is a URL. Domain: %s" % (word, domain))

        # Emoji count
        elif word in emoji.UNICODE_EMOJI:
            emojiCount += 1
            totalWordCount += -1
            if printTxt: print("Word %s is an emoji" % (word))
            idx = 'emoji_%s' % (word)
            twitOutputDict.setdefault(idx, 0)
            twitOutputDict[idx] += 1

            cleanedWordList.append(word)

        elif '#' in word:
            numOfHashtags += 1
            totalWordCount += -1

            twitOutputDict.setdefault('hash_%s' % (word), 0)
            twitOutputDict['hash_%s' % (word)] += 1

            cleanedWordList.append(word)

        else:
            # Don't count link length for avarage word length
            totalWordLength += len(word)

            for char in word:
                if char in string.punctuation:
                    punctuationDict.setdefault(char, 0)
                    punctuationDict[char] += 1
                    punctuationCount += 1
                if char.isupper():
                    totalUpperCase += 1

            for punc, countz in punctuationDict.items():
                twitOutputDict['pun_%s' % punc] = countz

            # strip word from punctuation:
            # if printTxt: print("Word Before: %s"%word)
            word = "".join([char.lower() for char in word if char not in punctuationDict.keys()]).strip()
            # if printTxt: print("Word After : %s"%word)

            # Too heavy and time consuming for a feature that wasn't very contributing - Removed
            # if word not in allWordsInEnglish_lower and not any(chr.isdigit() for chr in word):
            #   nonExistingWords+=1
            #   if printTxt: print('Word: %s does not exists in English'%(word))
            # else:
            #   wordLengthList.append(len(word))
            wordLengthList.append(len(word))

            if shouldCollectVocabularyWord and not isStopWord:
                twitOutputDict.update(vocabularyWordCatogy(word))

            # lemmatedWord = lemmatizer.lemmatize(word)
            stemmedWord = PorterStemmer().stem(word)
            # if printTxt: print(word,stemmedWord)

            if measureTime:
                timeFromStart = time() - start_time
                timeFromLast = time() - new_time
                new_time = time()
                print('Time from Start: %s, time from last: %s bow - PorterStemmer' % (
                round(timeFromStart), round(timeFromLast, 5)))

            if not isStopWord:
                twitOutputDict.setdefault('bow_%s' % (word), 0)
                twitOutputDict['bow_%s' % (word)] += 1

            cleanedWordList.append(word)
    if measureTime:
        timeFromStart = time() - start_time
        timeFromLast = time() - new_time
        new_time = time()
        print('Time from Start: %s, time from last: %s bow' % (round(timeFromStart), round(timeFromLast, 5)))

    ##############
    # String stats
    ##############

    if totalWordCount != 0:
        avarageLength = totalWordLength / totalWordCount
        upperCaseRatio = totalUpperCase / totalWordLength
        longestWordLength = max(wordLengthList)

    cleanedWordString = " ".join(cleanedWordList)

    twitOutputDict['aa_avarageLength'] = avarageLength
    twitOutputDict['aa_longestWordLength'] = longestWordLength
    twitOutputDict['aa_totalUpperCase'] = totalUpperCase
    twitOutputDict['aa_upperCaseRatio'] = upperCaseRatio
    twitOutputDict['aa_punctuationCount'] = punctuationCount
    twitOutputDict['aa_emojiCount'] = emojiCount
    twitOutputDict['aa_hasLink'] = hasLink
    twitOutputDict['aa_numOfHashtags'] = numOfHashtags
    twitOutputDict['aa_isCurseWordProbability'] = isCurseWordProbability
    twitOutputDict['aa_nonExistingWords'] = nonExistingWords
    twitOutputDict['aa_cleanedWordString'] = cleanedWordString

    ########
    # nGrams
    ########

    # Define function to iterate over word list and create Ngrams
    def ngrams(input, n, name):
        ngramDict = {}
        for i in range(len(input) - n + 1):
            nGram = input[i:i + n]
            nGram_str = '_'.join(nGram)
            indexName = '%s_%s' % (name, nGram_str)
            if indexName not in updatedStopWords:
                ngramDict.setdefault(indexName, 0)
                ngramDict[indexName] += 1

            if shouldCollectVocabularyWord:
                ngramDict.update(vocabularyWordCatogy(nGram_str))

        return ngramDict


    if len(cleanedWordList) > 1:
        biGram_dict = ngrams(cleanedWordList, 2, 'biGram')
        for head, count in biGram_dict.items():
            twitOutputDict.setdefault(head, 0)
            twitOutputDict[head] += count

        if len(cleanedWordList) > 2:
            triGram_dict = ngrams(cleanedWordList, 3, 'triGram')
            for head, count in triGram_dict.items():
                # print(head,count)
                twitOutputDict.setdefault(head, 0)
                twitOutputDict[head] += count

    if measureTime:
        timeFromStart = time() - start_time
        timeFromLast = time() - new_time
        new_time = time()
        print('Time from Start: %s, time from last: %s nGram' % (round(timeFromStart), round(timeFromLast, 5)))

    return twitOutputDict




def preprocessTweets(completeListOfPeopleAndTheirTweets,baseFolder='..\static\models\model' ,shouldProcessRawData=False):

    savePath_trainingData = os.path.join(baseFolder, 'Processed Tweet Data.csv')

    if shouldProcessRawData is False and os.path.exists(savePath_trainingData): #Load file if exists

        peopleAndTheirTweets_BOW = pd.read_csv(savePath_trainingData, header=0, index_col=0)

    else:

        peopleAndTheirTweets_BOW = {}
        listOfOriginalTweets = completeListOfPeopleAndTheirTweets['full_text']

        totalTweetCount = completeListOfPeopleAndTheirTweets.shape[0]

        listLength_ProgressBar = completeListOfPeopleAndTheirTweets.shape[0]
        bar = progressbar.ProgressBar(maxval=listLength_ProgressBar - 1,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        i = 0
        for item in completeListOfPeopleAndTheirTweets.head(listLength_ProgressBar).values:

            id_str = str(list(completeListOfPeopleAndTheirTweets.index)[i])

            item = item.tolist()

            full_text = item[0]
            if type(full_text)==int:
                item.remove(full_text)
                full_text = item[0]
            tweetDate = item[1]
            username = item[2].lower()
            fullName = item[3]
            acctdesc = item[4]
            location = item[5]
            following = item[6]
            followers = item[7]
            totaltweets = item[8]
            usercreatedts = item[9]
            tweetcreatedts = item[10]
            retweetcount = item[11]
            hashtags = item[12]
            truncated = False #int(item[13])
            userMentions = item[14]
            wordCountPerTwit = item[15]

            # if i == 0:
            #     print("\n", username, full_text, wordCountPerTwit, id_str)

            BOWDict = tweet_to_BOW(full_text, printTxt=False,vocabularyWord_Dict=vocabularyWord_Dict)

            BOWDict['aa_UserName'] = username
            BOWDict['aa_truncated'] = truncated
            BOWDict['aa_userMentions'] = userMentions
            BOWDict['aa_wordCountPerTwit'] = wordCountPerTwit

            #listOfOriginalTweets.append(full_text)

            peopleAndTheirTweets_BOW[id_str] = BOWDict
            bar.update(i)
            i += 1

    return peopleAndTheirTweets_BOW



#nltk.download('popular')



def removeFeaturesUnderThreshold(peopleAndTheirTweets_BOW, minThresholdForFeature=4, maxThresholdForFeature=1000, featureReduction_NumberOfFeaturesToLeave=10000):
    featuresAndTheirCount = {}

    for tweetId, tweet in peopleAndTheirTweets_BOW.items():
        twitHeaders = list(tweet.keys())
        # allHeaders.add(head for head in twitHeaders)
        for head in twitHeaders:
            featuresAndTheirCount.setdefault(head, 0)
            featuresAndTheirCount[head] += 1

    numFeaturesBefore = len(featuresAndTheirCount)

    print("\nNum Of Features before: %s" % (numFeaturesBefore))

    featuresAndTheirCount = {k: v for k, v in
                             sorted(featuresAndTheirCount.items(), reverse=True, key=lambda item: item[1])}


    print("\nMax Threshold: %s , MinThreshold: %s\n" % (str(maxThresholdForFeature), str(minThresholdForFeature)))

    headersToKeep = []
    headersExceedingUpperThreshold = []
    headersBelowLowerThreshold = []
    nonCountheaders = []

    maxReached = False
    for header, sumHeader in featuresAndTheirCount.items():
        if 'aa_' in header or 'url_' in header or 'pun_' in header or 'emoji_' in header or 'catg_' in header:
            headersToKeep.append(header)
            nonCountheaders.append(header)
            if len(headersToKeep) > featureReduction_NumberOfFeaturesToLeave:
                print("Maximum number of allowed features reached: %s" % len(headersToKeep))
                maxReached = True
                break
        elif sumHeader > maxThresholdForFeature:
            headersExceedingUpperThreshold.append(header)
        elif ('biGram_' in header and sumHeader < int(minThresholdForFeature / 2)) or (
                'triGram_' in header and sumHeader < int(minThresholdForFeature / 3)):
            headersBelowLowerThreshold.append(header)
        elif ('bow_' in header or 'hash_' in header) and sumHeader < minThresholdForFeature:
            headersBelowLowerThreshold.append(header)
        elif maxReached is False:
            headersToKeep.append(header)
            if len(headersToKeep) > featureReduction_NumberOfFeaturesToLeave:
                print("Maximum number of allowed features reached: %s" % len(headersToKeep))
                maxReached = True
                break

    nonCountheaders = list(set(nonCountheaders))

    headersToKeep.sort()

    listOfWordsToKeep = (
    'trump', 'donald''money''corona', 'coronavirus', 'fuck', 'shit', 'ass', 'bitch', 'hoe', 'fun', 'art',
    'donald trump', 'covid', 'read', 'shit', 'covid19', 'learning', 'video', 'country', 'america', 'together', 'work',
    'world', 'vote', 'president', 'gaga', 'black', 'lord', 'shall', 'thou', 'tesla', 'startship', 'new', 'model',
    'people')

    # updatedStopWords = [header.replace('bow_','') for header,sumHeader in featuresAndTheirCount.items() if 'bow_' in header and header not in headersToKeep]
    updatedStopWords = []  # = headersExceedingUpperThreshold
    for word in headersExceedingUpperThreshold:
        if word not in listOfWordsToKeep:
            updatedStopWords.append(word)
    for word in headersBelowLowerThreshold:
        if word not in listOfWordsToKeep:
            #updatedStopWords.append(word) #No need to add lower appearning words to stoplist
            pass
    # updatedStopWords.extend(headersBelowLowe rThreshold)

    updatedStopWords = [word.replace('bow_', '') for word in updatedStopWords]

    #pickle.dump(updatedStopWords, open(loadPretrainedStopwordList_Path, "wb"))

    headersBelowLowerThreshold_example = ['%s_%s' % (head, featuresAndTheirCount[head]) for head in
                                          headersBelowLowerThreshold]
    headersExceedingUpperThreshold_example = ['%s_%s' % (head, featuresAndTheirCount[head]) for head in
                                              headersExceedingUpperThreshold]

    print("Number of headersExceedingUpperThreshold: %s - Sample: %s\n" % (
    len(headersExceedingUpperThreshold), str(headersExceedingUpperThreshold_example[:10])))
    print("Number of headersBelowLowerThreshold: %s - Sample: %s\n" % (
    len(headersBelowLowerThreshold), str(headersBelowLowerThreshold_example[:10])))
    print("Number of non BOW/nGram related features: %s - Sample: %s\n" % (
    len(nonCountheaders), str(nonCountheaders[:10])))

    print("Number of stopWords collected: %s - Sample: %s\n" % (len(updatedStopWords), str(updatedStopWords[:10])))

    print("\nNumber of features to keep: %s" % (len(headersToKeep)))

    sleep(0.2)

    dictOfBOW_trimmed_new = {twitId: {} for twitId in peopleAndTheirTweets_BOW.keys()}

    count = 0
    listLength_ProgressBar = len(peopleAndTheirTweets_BOW)
    bar = progressbar.ProgressBar(maxval=listLength_ProgressBar - 1,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for tweetId_n, tweet_n in peopleAndTheirTweets_BOW.items():
        dictOfBOW_trimmed_new[tweetId_n] = {}
        tweetFeatures = list(tweet_n.keys())
        for feature in tweetFeatures:
            if feature in nonCountheaders:
                dictOfBOW_trimmed_new[tweetId_n][feature] = tweet_n[feature]
            elif feature in headersToKeep:
                dictOfBOW_trimmed_new[tweetId_n][feature] = tweet_n[feature]

        bar.update(count)
        count += 1

    featureTypesDistribution = {'Category': 0, 'Hardcoded': 0, 'URL': 0, 'Hashtag': 0, 'BOW': 0, 'BiGram': 0,
                                'TriGram': 0}

    for feature in headersToKeep:
        if 'catg_' in feature:
            featureTypesDistribution['Category'] += 1
        elif 'aa_' in feature:
            featureTypesDistribution['Hardcoded'] += 1
        elif 'url_' in feature:
            featureTypesDistribution['URL'] += 1
        elif 'hash_' in feature:
            featureTypesDistribution['Hashtag'] += 1
        elif 'bow_' in feature:
            featureTypesDistribution['BOW'] += 1
        elif 'biGram_' in feature:
            featureTypesDistribution['BiGram'] += 1
        elif 'triGram_' in feature:
            featureTypesDistribution['TriGram'] += 1

    print('\nfeatureTypesDistribution')
    print(featureTypesDistribution)

    print("\nNum Of Features After: %s" % (len(headersToKeep)))
    #print("\nHeaders to keep: %s" % (headersToKeep))

    #Save the updated stopword list
    updatedStopWords = saveLoadStopwords(updatedStopwordList=updatedStopWords, shouldLoadPretrainedStopwordList=False)

    return dictOfBOW_trimmed_new, updatedStopWords



def basic_preProcessing_BOW_FeatureRemoval(completeListOfPeopleAndTheirTweets, baseFolder_static, baseFolder_models, minThresholdForFeature=4, maxThresholdForFeature=1000, featureReduction_NumberOfFeaturesToLeave=15000, shouldPerformDataPreProcessingRegardless=False):

    listOfOriginalTweets = completeListOfPeopleAndTheirTweets['full_text']
    savePath_ProcessedData = os.path.join(baseFolder_models, 'Processed Data.csv')
    loadPretrainedStopwordList_Path = os.path.join(baseFolder_static, 'UpdatedStopWords.csv')

    if os.path.exists(savePath_ProcessedData) and os.path.exists(loadPretrainedStopwordList_Path) and not shouldPerformDataPreProcessingRegardless:
        print("Will load preprocessed data and stopwords from %s"%(loadPretrainedStopwordList_Path))
        peopleAndTheirTweets_df = pd.read_csv(savePath_ProcessedData, header=0, index_col=0)
        stopWordsList = saveLoadStopwords(baseFolder=baseFolder_static,
                                          shouldLoadPretrainedStopwordList=True,
                                          updatedStopwordList=[])
    else:

        if not os.path.exists(baseFolder_models):
            os.makedirs(baseFolder_models)

        stopWordsList = saveLoadStopwords(baseFolder=baseFolder_static,
                                          shouldLoadPretrainedStopwordList=False,
                                          updatedStopwordList=[])

        global vocabularyWord_Dict
        global categoryWord_Dict

        vocabularyWord_Dict,categoryWord_Dict = vocabularyWordList(baseFolder=baseFolder_static,
                                                                   #categoriesToExclude=categoriesToExclude,
                                                                   oveerideCurrentVoc=False,
                                                                   stopWordsList=stopWordsList)



        peopleAndTheirTweets_BOW = preprocessTweets(completeListOfPeopleAndTheirTweets=completeListOfPeopleAndTheirTweets,baseFolder=baseFolder_models ,shouldProcessRawData=False)

        peopleAndTheirTweets_BOW_trimmed, updatedStopWords = removeFeaturesUnderThreshold(peopleAndTheirTweets_BOW=peopleAndTheirTweets_BOW, minThresholdForFeature=minThresholdForFeature, maxThresholdForFeature=maxThresholdForFeature, featureReduction_NumberOfFeaturesToLeave=featureReduction_NumberOfFeaturesToLeave)

        peopleAndTheirTweets_df = pd.DataFrame(peopleAndTheirTweets_BOW_trimmed)
        peopleAndTheirTweets_df = peopleAndTheirTweets_df.T.fillna(0)
        headers = list(peopleAndTheirTweets_df.columns)
        headers.sort()
        peopleAndTheirTweets_df = peopleAndTheirTweets_df.filter(items=headers)

        peopleAndTheirTweets_df['full_tweet'] = listOfOriginalTweets.values


        peopleAndTheirTweets_df.to_csv(savePath_ProcessedData)
        print('Save processed data to: %s\n' % savePath_ProcessedData)

    return peopleAndTheirTweets_df,stopWordsList
