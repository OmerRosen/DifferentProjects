from gensim import matutils, models as gensin_models, corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import time
import pandas as pd
import os
from time import time
from nltk import pos_tag,word_tokenize

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import scipy.sparse
import pickle
import inspect
import progressbar
from textblob import TextBlob

#Required parameters:


def clean_df_text_from_nounsAndAdj(dataFrameWithText,textColumnName,applyStemming,minCountThreshold,maxCountThreshold,updatedStopWords=[]):

    #Recieves df with raw tweets (and tweetId as index),
    #Returns DF where coloumns are word features and index is tweet

  def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos) and word not in ["amp","co","https"]]
    valueToReturn = ' '.join(nouns_adj)
    if len(nouns_adj)<2: #If text without noun_adj is too small - use full text
      valueToReturn=text
      #print('Not enoughh nouns/adj found in : %s'%(text))
    return valueToReturn

  def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens if token not in updatedStopWords]
    return words

  if applyStemming:
    tokenizer=textblob_tokenizer
  else:
    tokenizer=None

  try:
    cleanedWordStringList_nounsAndAdj = pd.DataFrame(dataFrameWithText[textColumnName].apply(nouns_adj))
    #If the number of tweets is small - Set low min threshold
    numOfTweets = cleanedWordStringList_nounsAndAdj.shape[0]
    if numOfTweets<maxCountThreshold*2:
      print("Reducing min thresholds and stopwords because there are only %s tweets in the dataset"%(numOfTweets))
      min_df=0
      max_df=1000
      cvna = CountVectorizer(tokenizer=tokenizer)
    else:
      #cvna = CountVectorizer(tokenizer=tokenizer,stop_words=updatedStopWords) #min_df=min_df,max_df = max_df,
      cvna = CountVectorizer(min_df=minCountThreshold,max_df = maxCountThreshold,tokenizer=tokenizer,stop_words=updatedStopWords)
    data_cvna = cvna.fit_transform(cleanedWordStringList_nounsAndAdj[textColumnName])
    dynamicTopicModel = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
    dynamicTopicModel.index = cleanedWordStringList_nounsAndAdj.index
    listOfCorpusFeatures = list(dynamicTopicModel.T.index)

    print("Original Dataset shape: %s" % str(dynamicTopicModel.shape))

  except ValueError as e:
    functionName = inspect.currentframe().f_code.co_name
    print("Excepion on %s"%(functionName))
    #print(cleanedWordStringList_nounsAndAdj)
    print(e)

  except Exception as e:
    functionName = inspect.currentframe().f_code.co_name
    print("Excepion on %s"%(functionName))
    print(e)
    raise
  return dynamicTopicModel,listOfCorpusFeatures


def removeRedundentFeatures_toTDM(cleanedWordStringList_nounsAndAdj, minCountThreshold, maxCountThreshold,
                                  useStopWords,updatedStopWords):
    try:

        listOfFeaturesAndTheirCount = list(cleanedWordStringList_nounsAndAdj.sum().items())

        stop_words = []
        if useStopWords:
            stop_words = updatedStopWords

        featuresToDrop = []

        for feat, count in listOfFeaturesAndTheirCount:

            if count >= maxCountThreshold or count <= minCountThreshold or feat in stop_words:
                featuresToDrop.append(feat)

        print('df shape before: %s ' % (str(cleanedWordStringList_nounsAndAdj.shape)))
        print(len(featuresToDrop))

        termDocumentMatric = cleanedWordStringList_nounsAndAdj.drop(labels=featuresToDrop, axis=1).T

        print('df shape after : %s ' % (str(termDocumentMatric.shape)))

        #termDocumentMatric

    except Exception as e:
        functionName = inspect.currentframe().f_code.co_name
        print("Excepion on %s" % (functionName))
        print(e)
        print(listOfFeaturesAndTheirCount)
        print(feat, count)
        raise

    return termDocumentMatric

def createCorpusAndId2Word(termDocumentMatric,printSample=True):

  try:
    dictOfWords = termDocumentMatric.replace(0, np.nan, inplace=False)
    dictOfWords = {k: v[v.notna()].to_dict() for k,v in termDocumentMatric.items()}

    dict_termDocumentMatric = termDocumentMatric.to_dict()

    twitsAndTheirUniqueWordList = {}
    for twit,bow in termDocumentMatric.items():
      twitsAndTheirUniqueWordList[twit]=[]
      bowItems = list(bow.items())
      for feat, count in bowItems:
        if count!=0:
          twitsAndTheirUniqueWordList[twit].append(feat)
        else: #Corpus can't have empty list - If no value - Insert 'a'
            pass
      # Corpus can't have empty list - If no value - Insert 'a'
      if twitsAndTheirUniqueWordList[twit]==[]:
          twitsAndTheirUniqueWordList[twit].append('emptyWordReplacement')

    print(twitsAndTheirUniqueWordList.values())

    corpus_NounAdj = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(termDocumentMatric))
    id2word_nounAdj = corpora.Dictionary(documents=twitsAndTheirUniqueWordList.values())
    #id2word_nounAdj = {i:feat for i, feat in enumerate(termDocumentMatric.index)}

    if printSample:
      print(twitsAndTheirUniqueWordList[twit])
      sampleList = corpus_NounAdj[-2]
      sampleList = list((idx,id2word_nounAdj[idx],val) for idx,val in sampleList)
      print(sampleList)

  except Exception as e:
    functionName = inspect.currentframe().f_code.co_name
    print("Excepion on %s"%(functionName))
    print(e)
    raise

  return corpus_NounAdj,id2word_nounAdj,twitsAndTheirUniqueWordList


def train_Various_LDA_Models(corpus_NounAdj, id2word_nounAdj, listOfTwitsAndUniqueWords, runRegardless,
                             baseSavePath="/content/drive/MyDrive/Harvard HW/Course 2 - Final Project/omertest/",
                             overrideTrainSettings=True):
    if type(overrideTrainSettings) == dict:  # If empty, use combination below
        print('overrideTrainSettings, will use ONLY set values of 4 Topics + 10 Passes')
        numTopics = overrideTrainSettings['numTopics']
        numOfPasses = overrideTrainSettings['numOfPasses']
    else:
        numTopics = [8]
        numOfPasses = [2]
    print('Will test on %s topics, %s passes' % (numTopics, numOfPasses))

    outputSavePath = os.path.join(baseSavePath, "LDA_Topic_Model_Output.csv")
    runRegardless = True

    if not os.path.exists(outputSavePath) or runRegardless:
        if not os.path.exists(baseSavePath):
            os.makedirs(baseSavePath)

        ldaResultOutput = {}
        runCount = 0
        for top in numTopics:
            for passN in numOfPasses:
                runCount+=1

                ldaModelTitle = '\nLDA_%s_Topics_%s_Passes - RunCount: %s' % (top, passN,runCount)
                print(ldaModelTitle)
                # Start the clock:
                start_time = time()

                ldaResultOutput[ldaModelTitle] = {'TopicNum': top, 'PassNum': passN}

                lda_nounAdj = gensin_models.LdaModel(corpus=corpus_NounAdj, num_topics=top, passes=passN,
                                                     id2word=id2word_nounAdj,iterations=100)

                print("Created gensin_models.LdaModel")
                # perplexity
                Perplexity = lda_nounAdj.log_perplexity(corpus_NounAdj)

                # coherence score
                coherence_model = CoherenceModel(model=lda_nounAdj, texts=listOfTwitsAndUniqueWords.values(),
                                            dictionary=id2word_nounAdj, coherence='c_v',processes=1) ## processes must be 1 or else freeze issues

                print("Created CoherenceModel")
                try:
                    coherence = coherence_model.get_coherence()
                except Exception as e:
                    print("Exception when running coherence_model.get_coherence():")
                    print(e)
                    coherence=0

                ldaResultOutput[ldaModelTitle]['TopicNum'] = top
                ldaResultOutput[ldaModelTitle]['PassNum'] = passN
                ldaResultOutput[ldaModelTitle]['Perplexity'] = Perplexity
                ldaResultOutput[ldaModelTitle]['Coherence'] = coherence

                print('Num of topics: %s | Num of passes: %s | Perplexity: %s | Coherence Score: %s' % (
                top, passN, Perplexity, coherence))
                timeInSeconds = time() - start_time
                print('Finshed training %s in %s seconds.\n' % (ldaModelTitle, int(timeInSeconds)))

        ldaResultOutput_df = pd.DataFrame(ldaResultOutput).T.sort_values(by=['Perplexity'], ascending=True)

        ldaResultOutput_df.to_csv(outputSavePath)

    return ldaResultOutput_df


def loadAndTrainSelectedModel(corpus_NounAdj,id2word_nounAdj,baseSavePath="/content/drive/MyDrive/Harvard HW/Course 2 - Final Project/omertest/",trainNewModelRegardless=False,printResults=False):

  outputSavePath = os.path.join(baseSavePath,"LDA_Topic_Model_Output.csv")
  ldaResultOutput_df = pd.read_csv(outputSavePath,header=0,index_col=0)

  #ldaResultOutput_df = ldaResultOutput_df.sort_values(by=['Perplexity'],ascending=True)
  ldaResultOutput_df = ldaResultOutput_df.sort_values(by=['Coherence'],ascending=False)

  modelName = list(ldaResultOutput_df.index)[0].strip()
  best_topicNum = int(ldaResultOutput_df.values[0][0])
  best_passNum = int(ldaResultOutput_df.values[0][1])

  print("Selected LDA model: %s %s topics, %s passes"%(modelName,best_topicNum,best_passNum))
  modelSavePath = os.path.join(baseSavePath,'%s'%(modelName))

  if best_topicNum==0:
    print("best_topicNum was 0 for some reason.")
    best_topicNum=4

  if os.path.exists(modelSavePath) and not trainNewModelRegardless:
    print("Model %s already exists"%(modelName))
    lda_nounAdj = gensin_models.LdaModel.load(modelSavePath)
    print("Model %s was loaded from %s"%(modelName,modelSavePath))
  else:
    lda_nounAdj = gensin_models.LdaModel(corpus=corpus_NounAdj,num_topics=best_topicNum,passes=best_passNum,id2word=id2word_nounAdj)
    lda_nounAdj.save(modelSavePath)


    print("Model %s was saved to %s"%(modelName,modelSavePath))

  top20WordsPerTopic = {}

  for topinId in range(best_topicNum):
    wordIdPerTopic = lda_nounAdj.get_topic_terms(topinId, topn=1000)
    #print(wordIdPerTopic)
    wordPerTopic = [id2word_nounAdj[word] for word,stragth in wordIdPerTopic]
    if printResults:
      print("TopicId %s"%(topinId),wordPerTopic)
    top20WordsPerTopic["TopicId %s"%(topinId)] = wordPerTopic
  try:
    pass

  except Exception as e:
    functionName = inspect.currentframe().f_code.co_name
    print("Excepion on %s"%(functionName))
    print(e)
    raise
  return lda_nounAdj,top20WordsPerTopic,ldaResultOutput_df,best_topicNum


def assign_Topic_To_document(selectedLDAModel,corpus_NounAdj,id2word_nounAdj,dataFrameWithText,textColumnName,printResults=False):
  topicAssociationList = selectedLDAModel.get_document_topics(corpus_NounAdj)
  twitIdList = list(dataFrameWithText.index)
  twitTextList = list(dataFrameWithText[textColumnName])

  topicDict = {}

  if printResults:
    print(len(topicAssociationList))
    print(topicAssociationList)
  for i,topisAss in enumerate(topicAssociationList):
    twitIfd = twitIdList[i]
    twitTxt = twitTextList[i]
    topicDict[twitIfd] = {'twitTxt':twitTxt}
    if i==2 and printResults:
      print([(id2word_nounAdj[item[0]],item[1]) for item in corpus_NounAdj[i]])
      print(twitIfd,topisAss,twitTxt)
      print("\n")

    for topId,probability in topisAss:
      idxName = 'topic_%s'%(topId)
      topicDict[twitIfd][idxName] = probability

  return topicDict


class LDA_Perdictions():

    def __init__(self, baseSavePath, extandedProjectName):
        print("Initiated LDA model. \nRun function main_LDA_function_textToTopics to train a new model.")
        self.data_dtmna = None
        self.termDocumentMatric = None
        self.corpus_NounAdj = None
        self.id2word_nounAdj = None
        self.lda_nounAdj = None

        self.baseSavePath = baseSavePath
        self.trainingResultsPath = os.path.join(baseSavePath, "LDA_Topic_Model_Output.csv")
        self.path_LDAModel = os.path.join(self.baseSavePath, extandedProjectName + "_LDAModel.pkl")
        self.path_Id2Words = os.path.join(self.baseSavePath, extandedProjectName + "_Id2Words.pkl")
        self.path_Corpus = os.path.join(self.baseSavePath, extandedProjectName + "__Corpus.pkl")
        self.path_mainCorpusFeatures = os.path.join(self.baseSavePath, extandedProjectName + "_mainCorpusFeatures.pkl")

        self.extandedProjectName = extandedProjectName

    def main_LDA_function_textToTopics(self, dataFrameWithText, textColumnName, applyStemming=False,
                                       minCountThreshold=0, maxCountThreshold=10, useStopWords=True,
                                       trainNewModelRegardless=True, trainSeveralLDAModels=False,
                                       overrideTrainSettings={'numTopics':[5,7,10],'numOfPasses':[2,5]}, printResults=False,
                                       updatedStopWords=[]
                                       ):

        # Remember Parameters:
        self.dataFrameWithText = dataFrameWithText
        self.textColumnName = textColumnName
        self.applyStemming = applyStemming
        self.minCountThreshold = minCountThreshold
        self.maxCountThreshold = maxCountThreshold
        self.useStopWords = useStopWords
        self.trainNewModelRegardless = trainNewModelRegardless
        self.trainSeveralLDAModels = trainSeveralLDAModels
        self.overrideTrainSettings = overrideTrainSettings
        self.numberOfTopics = None
        self.updatedStopWords = updatedStopWords

        try:

            if not (os.path.exists(self.trainingResultsPath)
                    and os.path.exists(self.path_LDAModel)
                    and os.path.exists(self.path_Id2Words)
                    and os.path.exists(self.path_Corpus)
                    and os.path.exists(self.path_mainCorpusFeatures)
                    and not trainNewModelRegardless):  # If model was already trained, skip data preparation stages and load it instead

                # Take raw text and remove all none-nouns and adj, convert to BOW:
                # if self.data_dtmna is None:
                data_dtmna = clean_df_text_from_nounsAndAdj(dataFrameWithText, textColumnName, applyStemming,minCountThreshold=self.minCountThreshold,maxCountThreshold=self.maxCountThreshold,updatedStopWords=self.updatedStopWords)
                self.data_dtmna = data_dtmna

                # Take BOW, remove redundent features, convert to TDM (Transpose)
                # if self.termDocumentMatric is None:
                termDocumentMatric = removeRedundentFeatures_toTDM(cleanedWordStringList_nounsAndAdj=self.data_dtmna,
                                                                   minCountThreshold=minCountThreshold,
                                                                   maxCountThreshold=maxCountThreshold,
                                                                   useStopWords=useStopWords,
                                                                   updatedStopWords=self.updatedStopWords)
                self.termDocumentMatric = termDocumentMatric

                # Take TDM, convert to corpus and index words
                corpus_NounAdj, id2word_nounAdj, listOfTwitsAndUniqueWords = createCorpusAndId2Word(
                                                                    termDocumentMatric=self.termDocumentMatric)

                # If needed, train various LDA Models
                ldaResultOutput_df = train_Various_LDA_Models(corpus_NounAdj=corpus_NounAdj,
                                                              id2word_nounAdj=id2word_nounAdj,
                                                              listOfTwitsAndUniqueWords=listOfTwitsAndUniqueWords,
                                                              runRegardless=trainSeveralLDAModels,
                                                              overrideTrainSettings=overrideTrainSettings,
                                                              baseSavePath=self.baseSavePath)

                # Take corpus and use it to train LDA model
                print("ldaResultOutput_df:")
                print(ldaResultOutput_df)
                lda_nounAdj, top20WordsPerTopic, ldaResultOutput_df, numberOfTopics = loadAndTrainSelectedModel(
                                                                corpus_NounAdj=corpus_NounAdj,
                                                                id2word_nounAdj=id2word_nounAdj,
                                                                trainNewModelRegardless=trainNewModelRegardless,
                                                                baseSavePath=self.baseSavePath,
                                                                printResults=printResults)

                self.data_dtmna = data_dtmna
                self.termDocumentMatric = termDocumentMatric
                self.listOfTwitsAndUniqueWords = listOfTwitsAndUniqueWords
                self.ldaResultOutput_df = ldaResultOutput_df
                self.mainCorpusFeatures = list(self.termDocumentMatric.index)
                self.top20WordsPerTopic = top20WordsPerTopic
                self.dataFrameWithText = dataFrameWithText
                self.numberOfTopics = numberOfTopics

                self.corpus_NounAdj = corpus_NounAdj
                self.id2word_nounAdj = id2word_nounAdj
                self.lda_nounAdj = lda_nounAdj

                listOfColors = ['Brown', 'Chocolate', 'Blue', 'BlueViolet', 'DarkOliveGreen', 'DarkOrchid',
                                'DarkSlateGrey', 'LightBlue', 'Gold', 'Fuchsia', 'DarkSlateGray']

                self.listOfColors = {i: color for i, color in enumerate(listOfColors)}



            else:
                print("Skip data preperation and load pre-trained model froma  pickle file: %s" % self.baseSavePath)

                self.load_model_Pikcle()

                outputSavePath = self.baseSavePath + "LDA_Topic_Model_Output.csv"
                ldaResultOutput_df = pd.read_csv(outputSavePath, header=0, index_col=0)

                ldaResultOutput_df = ldaResultOutput_df.sort_values(by=['Coherence'], ascending=False)

                modelName = list(ldaResultOutput_df.index)[0].strip()
                best_topicNum = int(ldaResultOutput_df.values[0][0])
                if best_topicNum == 0: best_topicNum = 4
                best_passNum = int(ldaResultOutput_df.values[0][1])

                self.ldaResultOutput_df = ldaResultOutput_df
                self.numberOfTopics = best_topicNum

            # Make prediction using LDA:
            topicDict = assign_Topic_To_document(self.lda_nounAdj, self.corpus_NounAdj, self.id2word_nounAdj,
                                                 self.dataFrameWithText, textColumnName=self.textColumnName,
                                                 printResults=printResults)
            self.topicDict = topicDict

            # Set dictionary for both Topics and Words in topics

            model_topics = self.lda_nounAdj.show_topics(num_words=20000, formatted=False)
            print("Model topics:")
            print(model_topics)
            topics_And_Their_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in model_topics]
            print("topics_And_Their_words:")
            print(topics_And_Their_words)
            listOfColors = ['Brown', 'Chocolate', 'Blue', 'BlueViolet', 'DarkOliveGreen', 'DarkOrchid', 'DarkSlateGrey',
                            'LightBlue', 'Gold', 'Fuchsia', 'DarkSlateGray']

            self.listOfColors = {i: color for i, color in enumerate(listOfColors)}

            wordsAndTheirTopics = {}
            for topicId, tupleOfWords in model_topics:
                for word, strangth in tupleOfWords:
                    if word not in updatedStopWords:
                        wordsAndTheirTopics.setdefault(word, {})
                        wordsAndTheirTopics[word].setdefault('strangth', 0)
                        if strangth > wordsAndTheirTopics[word]['strangth']:
                            wordsAndTheirTopics[word]['topic'] = topicId
                            wordsAndTheirTopics[word]['strangth'] = strangth
                            try:
                                wordsAndTheirTopics[word]['color'] = listOfColors[topicId]
                            except:
                                wordsAndTheirTopics[word]['color'] = 'DarkSlateGray'

            self.model_topics = model_topics
            self.topics_And_Their_words = topics_And_Their_words
            self.wordsAndTheirTopics = wordsAndTheirTopics

            self.save_model_Pikcle()

            return topicDict, self.lda_nounAdj, wordsAndTheirTopics

        except Exception as e:
            functionName = inspect.currentframe().f_code.co_name
            print("Excepion on %s" % (functionName))
            print(e)
            raise


    def save_model_Pikcle(self):
        pickle.dump(self.lda_nounAdj, open(self.path_LDAModel, "wb"))
        pickle.dump(self.corpus_NounAdj, open(self.path_Corpus, "wb"))
        pickle.dump(self.id2word_nounAdj, open(self.path_Id2Words, "wb"))
        pickle.dump(self.mainCorpusFeatures, open(self.path_mainCorpusFeatures, "wb"))

        print('Saved data to %s' % (self.baseSavePath + "LDAModel.pkl"))

    def load_model_Pikcle(self):
        self.lda_nounAdj = pickle.load(open(self.path_LDAModel, "rb"))
        self.corpus_NounAdj = pickle.load(open(self.path_Corpus, "rb"))
        self.id2word_nounAdj = pickle.load(open(self.path_Id2Words, "rb"))
        self.mainCorpusFeatures = pickle.load(open(self.path_mainCorpusFeatures, "rb"))

    def predict_On_Unseen_Corpus(self, new_dataFrameWithText):

        # Take raw text and remove all none-nouns and adj, convert to BOW:
        new_data_dtmna = clean_df_text_from_nounsAndAdj(new_dataFrameWithText, self.textColumnName, self.applyStemming,
                                                        minCountThreshold=self.minCountThreshold,maxCountThreshold=self.maxCountThreshold)

        # Take BOW, remove redundent features, convert to TDM (Transpose)
        print("Number of feature of new dataset: %s" % (str(new_data_dtmna.shape)))
        new_termDocumentMatric = new_data_dtmna.filter(
            items=self.mainCorpusFeatures).T  # filter(items=self.mainCorpusFeatures,axis=1).
        print("Number of feature of new dataset: %s" % (str(new_termDocumentMatric.shape)))

        self.new_dataFrameWithText = new_dataFrameWithText
        self.new_termDocumentMatric = new_termDocumentMatric

        new_Corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(new_termDocumentMatric))

        # for i,corp in enumerate(new_Corpus):
        #   print(i,corp)

        self.new_Corpus = new_Corpus

        topicDict_new = assign_Topic_To_document(self.lda_nounAdj,  # Use previsouly trained LDA model
                                                 new_Corpus,  # Use the new corpus
                                                 self.id2word_nounAdj,  # Use existing Id2Word dictionary
                                                 new_dataFrameWithText,
                                                 self.textColumnName)  # Use new dataFrameWithText

        self.topicDict_new = topicDict_new
        self.new_Corpus = new_Corpus

        return topicDict_new, new_Corpus

    def print_corpus(self, corpus, numOfTxts):

        for i, corp in enumerate(corpus):
            if i <= numOfTxts:
                print(corpus)
                print([(self.id2word_nounAdj[item[0]], item[1]) for item in corpus[i]])

    def plot_pyLDAvis(self):
        import pyLDAvis.gensim
        import pickle
        import pyLDAvis
        # Visualize the topics
        pyLDAvis.enable_notebook()
        LDAvis_prepared = pyLDAvis.gensim.prepare(self.lda_nounAdj, self.corpus_NounAdj, self.id2word_nounAdj)
        return LDAvis_prepared

    def convert_string_into_HTML(self, text, wordsAndTheirTopics=None, fontSize_px=None):

        listOfColors = ['Brown', 'Chocolate', 'Blue', 'BlueViolet', 'DarkOliveGreen', 'DarkOrchid', 'DarkSlateGrey',
                        'LightBlue', 'Gold', 'Fuchsia', 'DarkSlateGray']

        fontFeat = ""
        if fontSize_px:
            fontFeat = "font-size:%spx" % (fontSize_px)

        listOfWords = text.split()

        if self.applyStemming:
            blob = TextBlob(text.lower())
            tokens = blob.words
            listOfWords_stemmed = [token.stem() for token in tokens]
        else:
            listOfWords_stemmed = listOfWords

        if len(listOfWords) != len(listOfWords_stemmed):
            listOfWords_stemmed = listOfWords

        if wordsAndTheirTopics is None:
            wordsAndTheirTopics = self.wordsAndTheirTopics

        html_string = "<p>"
        for i, word in enumerate(listOfWords_stemmed):
            originalWord = listOfWords[i]
            if word in self.wordsAndTheirTopics.keys():
                html_p = '<span  style="color:%s;%s">%s </span>' % (
                wordsAndTheirTopics[word]['color'], fontFeat, originalWord)
            else:
                html_p = '%s ' % (originalWord)

            html_string += html_p

        return html_string + "</p>"


def transalte_corpusIdx_toString(corpusRow,id2word):
    translatedRow = list((idx, id2word[idx], val) for idx, val in corpusRow)
    return  translatedRow


def dynamicTopicModel_ToCorpus(dynamicTopicModel):
    dynamicTopicModel_dict = dynamicTopicModel.T.to_dict()

    twitsAndTheirUniqueWordList = {}
    for twit, values in dynamicTopicModel_dict.items():
        twitsAndTheirUniqueWordList[twit]=[]
        for col,count in values.items():
            if count>0:
                twitsAndTheirUniqueWordList[twit].append(col)


    # #Example how it should look
    # twitsAndTheirUniqueWordList_Token = [
    #     ['human', 'interface', 'computer'],
    #     ['survey', 'user', 'computer', 'system', 'response', 'time'],
    #     ['user', 'response', 'time'],
    #     ['trees']
    #     ]

    twitsAndTheirUniqueWordList_Token = list(twitsAndTheirUniqueWordList.values())

    dictionaryForLDA = id2word = Dictionary(twitsAndTheirUniqueWordList_Token)
    corpus = [dictionaryForLDA.doc2bow(text) for text in twitsAndTheirUniqueWordList_Token]

    return corpus, id2word, twitsAndTheirUniqueWordList_Token

def takeTokenList_ReturnModel(tokenList, dictionaryForLDA, corpus, baseFolder, topicList, passList, loadTrainedLDAIfExists):

    winningModel_SavePath = os.path.join(baseFolder,'Winning LDA Model')
    path_LDA_LTrainingOutput = os.path.join(baseFolder, "LDA_LTrainingOutput.csv")
    if loadTrainedLDAIfExists and os.path.exists(winningModel_SavePath) and os.path.exists(path_LDA_LTrainingOutput):
        print ("Loading pre-trained LDA model from %s"%(winningModel_SavePath))
        winningLDAModel = gensin_models.LdaModel.load(winningModel_SavePath)
        ldaResultOutput_df = pd.read_csv(path_LDA_LTrainingOutput, header=0, index_col=0)
        _, numberOfTopics = pd.DataFrame(ldaResultOutput_df).sort_values(by=['Coherence'], ascending=False).filter(items=['ActualModel', 'TopicNum']).head(1).values[0]

    else:
        if type(topicList)==int:
            topicList=[topicList]
        elif type(topicList)==list:
            topicList=topicList
        else:
            topicList=[7]
        print ('LDA Topis to check: %s'%str(topicList))

        if type(passList)==int:
            passList=[passList]
        elif type(passList)==list:
            passList=passList
        else:
            passList=[10]

        print('LDA Passes to check: %s' % str(passList))

        ldaResultOutput={}

        for top in topicList:
            for passN in passList:

                ldaModelTitle = '\nLDA_%s_Topics_%s_Passes' % (top, passN)
                start_time = time()
                print("Training LDA Model: %s - StartTime: %s"%(ldaModelTitle,start_time))

                ldaResultOutput[ldaModelTitle] = {'TopicNum': top, 'PassNum': passN}

                ldaTest = LdaModel(corpus=corpus, id2word=dictionaryForLDA, iterations=100, num_topics=top, passes=passN)
                Perplexity = ldaTest.log_perplexity(corpus)

                cohrM = CoherenceModel(model=ldaTest, texts=tokenList, corpus=corpus,  dictionary=dictionaryForLDA, coherence='c_v', processes=1)
                cohrScore = cohrM.get_coherence()

                timeInSeconds = time() - start_time

                print("Coherence: %s"%(round(cohrScore,3)))

                ldaResultOutput[ldaModelTitle]['TopicNum'] = top
                ldaResultOutput[ldaModelTitle]['PassNum'] = passN
                ldaResultOutput[ldaModelTitle]['Perplexity'] = round(Perplexity,3)
                ldaResultOutput[ldaModelTitle]['Coherence'] = round(cohrScore,3)
                ldaResultOutput[ldaModelTitle]['TimeInSec'] = round(timeInSeconds,3)

                ldaResultOutput[ldaModelTitle]['ActualModel'] = ldaTest

        ldaResultOutput_df = pd.DataFrame(ldaResultOutput).T.sort_values(by=['Coherence'], ascending=False).copy()
        print(pd.DataFrame(ldaResultOutput_df).sort_values(by=['Coherence'], ascending=False))

        winningLDAModel,numberOfTopics,Coherence,Perplexity = pd.DataFrame(ldaResultOutput_df).sort_values(by=['Coherence'], ascending=False).filter(items=['ActualModel','TopicNum','Coherence','Perplexity']).head(1).values[0]



        #pickle.dump(winningLDAModel, open(winningModel_SavePath, "wb"))
        winningLDAModel.save(winningModel_SavePath)
        print("Winning Model Details:")
        print(ldaResultOutput_df.head(1).values)

    return winningLDAModel,ldaResultOutput_df,numberOfTopics,Coherence,Perplexity



def takeLADModel_ReturnTopicsBreakdown(winningModel,num_words_per_topic):
    model_topics = winningModel.show_topics(num_topics=-1,num_words=num_words_per_topic, formatted=False)
    print("Model topics:")
    print(model_topics)
    topics_And_Their_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in model_topics]
    print("topics_And_Their_words:")
    print(topics_And_Their_words)
    listOfColors = ['Brown', 'Chocolate', 'Blue', 'BlueViolet', 'DarkOliveGreen', 'DarkOrchid', 'DarkSlateGrey',
                    'LightBlue', 'Gold', 'Fuchsia', 'DarkSlateGray']
    listOfColors = {i: color for i, color in enumerate(listOfColors)}

    wordsAndTheirTopics = {}
    for topicId, tupleOfWords in model_topics:
        for word, strangth in tupleOfWords:
            wordsAndTheirTopics.setdefault(word, {})
            wordsAndTheirTopics[word].setdefault('strangth', 0)
            if strangth > wordsAndTheirTopics[word]['strangth']:
                wordsAndTheirTopics[word]['topic'] = topicId
                wordsAndTheirTopics[word]['strangth'] = strangth
                try:
                    wordsAndTheirTopics[word]['color'] = listOfColors[topicId]
                except:
                    wordsAndTheirTopics[word]['color'] = 'DarkSlateGray'


    return model_topics, topics_And_Their_words, wordsAndTheirTopics



def assign_Topic_To_document(preProcessedData_BOW_df,winningModel, corpus, id2word, numberOfTopics, dataFrameWithText, textColumnName, printResults=True):
    topicAssociationList = winningModel.get_document_topics(corpus)
    twitIdList = list(dataFrameWithText.index)
    twitTextList = list(dataFrameWithText[textColumnName])

    topicDict = {}
    topicHeaders = set()

    if printResults:
        print(len(topicAssociationList))
        print(topicAssociationList)
        print("Assigning topic probability to tweets:")
    i=0
    for topisAss in topicAssociationList:
        #print(i)

        try:
            if i == 154:
                print(i, twitIdList[i], twitTextList[i])
            twitIfd = twitIdList[i]
            twitTxt = twitTextList[i]
            topicDict[twitIfd] = {'twitTxt':twitTxt}

            if i==155 and printResults:
              print([(id2word[item[0]], item[1]) for item in corpus[i]])
              print(twitIfd,topisAss,twitTxt)
              print("\n")

            for topId,probability in topisAss:
              idxName = 'topic_%s'%(topId)
              topicDict[twitIfd][idxName] = probability
              topicHeaders.add(idxName)
        except Exception as e:
            print(e)

        i+=1
    for n in range(numberOfTopics):
        try:
            preProcessedData_BOW_df['topic_%s' % (n)] = None
        except Exception as e:
            print(e)


    print("for twitId, topics in topicDict.items():")
    for twitId, topics in topicDict.items():
        for topicHead, probability in topics.items():
            if 'topic' in topicHead:
                preProcessedData_BOW_df.loc[twitId, topicHead] = float(probability)
                topicHeaders.add(topicHead)


    headerToShow = list(topicHeaders)
    headerToShow.extend(['aa_UserName', 'full_tweet'])
    preProcessedData_BOW_df = preProcessedData_BOW_df.fillna(0)
    preProcessedData_BOW_df.filter(items=headerToShow).head(5)


    return preProcessedData_BOW_df


if __name__ == "__main__":

    minCountThreshold=4
    maxCountThreshold=450
    shouldTrainNewLDAModelRegardless=True
    topicList = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20]
    passList = [10]
    loadTrainedLDAIfExists = True

    baseFolder_models_test = r'C:\Users\omerro\Google Drive\Data Science Projects\OmersProjects\omerprojects\TwitterClassifierFolder\models\FinalProject_7Classes_20twtpp_authors_Jo_la_IA_ia_El_St_ka'
    test_processedData = os.path.join(baseFolder_models_test,'Processed Data.csv')
    test_updatedStopwords = r'C:\Users\omerro\Google Drive\Data Science Projects\OmersProjects\omerprojects\TwitterClassifierFolder\static\UpdatedStopWords.csv'

    preProcessedData_BOW_df = pd.read_csv(test_processedData, index_col=0)
    updatedStopWords = pickle.load(open(test_updatedStopwords, "rb"))

    listOfRawTweets_df = pd.DataFrame(preProcessedData_BOW_df['full_tweet'])

    dynamicTopicModel = clean_df_text_from_nounsAndAdj(dataFrameWithText=listOfRawTweets_df,
                                                       textColumnName='full_tweet',
                                                       applyStemming=False,
                                                       minCountThreshold=0,
                                                       maxCountThreshold=1000)


    corpus, id2word, twitsAndTheirUniqueWordList = dynamicTopicModel_ToCorpus(dynamicTopicModel)

    winningLDAModel,_,numberOfTopics = takeTokenList_ReturnModel(tokenList=twitsAndTheirUniqueWordList,
                                                dictionaryForLDA=id2word,
                                                corpus=corpus,
                                                baseFolder=baseFolder_models_test,
                                                topicList=topicList,
                                                passList=passList,
                                                loadTrainedLDAIfExists=loadTrainedLDAIfExists)