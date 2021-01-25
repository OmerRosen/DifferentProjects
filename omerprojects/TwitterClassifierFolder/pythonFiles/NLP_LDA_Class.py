from omerprojects.TwitterClassifierFolder.pythonFiles.NLP_LDA_Functions import clean_df_text_from_nounsAndAdj,\
    takeTokenList_ReturnModel,dynamicTopicModel_ToCorpus,\
    takeLADModel_ReturnTopicsBreakdown,assign_Topic_To_document

from gensim import models as gensin_models

import os
import pandas as pd
import pickle
import inspect
import pyLDAvis
import pyLDAvis.gensim

from textblob import TextBlob

class LDA_Perdictions():

    def __init__(self, baseFolder_models):
        print("Initiated LDA model. \nRun function main_LDA_function_textToTopics to train a new model.")

        #Create paths for all future saved items
        self.baseSavePath = os.path.join(baseFolder_models,'LDA')
        self.path_WordsToTopicList = os.path.join(self.baseSavePath, "WordsToTopicList.csv")
        self.path_LDAModel = os.path.join(self.baseSavePath, "LDA_Model.pkl")
        self.path_Id2Words = os.path.join(self.baseSavePath, "LDA_Id2Words.pkl")
        self.path_Corpus = os.path.join(self.baseSavePath, "LDA_Corpus.pkl")
        self.path_mainCorpusFeatures = os.path.join(self.baseSavePath, "LDA_mainCorpusFeatures.pkl")
        self.path_LDATrainingOutput = os.path.join(self.baseSavePath, "LDA_TrainingOutput.csv")
        self.num_words_per_topic = 1000
        self.loadTrainedLDAIfExists = False
        self.textColumnName = 'full_tweet'
        self.applyStemming = False

        #Parameters to be configured later on:
        self.corpus = None
        self.id2word = None
        self.winningLDAModel = None
        self.twitsAndTheirUniqueWordList = None
        self.winningLDAModel = None

        listOfColors = ['Brown', 'Chocolate', 'Blue', 'BlueViolet', 'DarkOliveGreen', 'DarkOrchid', 'DarkSlateGrey',
                        'LightBlue', 'Gold', 'Fuchsia', 'DarkSlateGray']
        self.listOfColors = {i: color for i, color in enumerate(listOfColors)}

        if not os.path.exists(self.baseSavePath):
            os.makedirs(self.baseSavePath)


    def main_LDA_function_textToTopics(self, dataFrameWithText, textColumnName, applyStemming=False,
                                       minCountThreshold=0, maxCountThreshold=10, useStopWords=True,
                                       trainNewLDAModelRegardless=False, trainSeveralLDAModels=False,
                                       topicList=[3,4,5,7,9], printResults=False

                                       ):

        # Remember Parameters:
        self.dataFrameWithText = dataFrameWithText
        self.textColumnName = textColumnName
        self.applyStemming = applyStemming
        self.minCountThreshold = minCountThreshold
        self.maxCountThreshold = maxCountThreshold
        self.useStopWords = useStopWords
        self.trainNewModelRegardless = trainNewLDAModelRegardless
        self.trainSeveralLDAModels = trainSeveralLDAModels
        self.topicList = topicList
        self.numberOfTopics = None
        self.printResults = printResults



        try:

            if not (    os.path.exists(self.path_LDATrainingOutput)
                    and os.path.exists(self.path_WordsToTopicList)
                    and os.path.exists(self.path_LDAModel)
                    and os.path.exists(self.path_Id2Words)
                    and os.path.exists(self.path_Corpus)
                    and os.path.exists(self.path_mainCorpusFeatures)
                    and not trainNewLDAModelRegardless):  # If model was already trained, skip data preparation stages and load it instead

                # Take raw text and remove all none-nouns and adj, convert to BOW:
                # if self.data_dtmna is None:

                self.termDocumentMatric,self.mainCorpusFeatures = clean_df_text_from_nounsAndAdj(dataFrameWithText,
                                           applyStemming=False,
                                           minCountThreshold=0,
                                           maxCountThreshold=1000,
                                           textColumnName='full_tweet')


                # Take termDocumentMatric, convert to corpus and index words
                self.corpus, self.id2word, self.twitsAndTheirUniqueWordList = dynamicTopicModel_ToCorpus(self.termDocumentMatric)

                # If needed, train various LDA Models
                self.winningLDAModel, self.ldaResultOutput_df, self.numberOfTopics,self.Coherence,self.Perplexity = takeTokenList_ReturnModel(tokenList=self.twitsAndTheirUniqueWordList,
                                                            dictionaryForLDA=self.id2word,
                                                            corpus=self.corpus,
                                                            baseFolder=self.baseSavePath,
                                                            topicList=self.topicList,
                                                            passList=[10],
                                                            loadTrainedLDAIfExists=self.loadTrainedLDAIfExists)

                # Take corpus and use it to train LDA model
                if printResults:
                    print("lda winningLDAModel: (%s topics)"%(self.numberOfTopics))
                    print(self.winningLDAModel)

                self.model_topics, self.topics_And_Their_words, self.wordsAndTheirTopics = takeLADModel_ReturnTopicsBreakdown(self.winningLDAModel,self.num_words_per_topic)

                if printResults:
                    print("self.model_topics")
                    print(self.model_topics)

                self.save_model_Pikcle()

            else:
                print("Skip data preperation and load pre-trained model from a  pickle file: %s" % self.baseSavePath)

                self.load_model_Pikcle()


        except Exception as e:
            functionName = inspect.currentframe().f_code.co_name
            print("Excepion on %s" % (functionName))
            print(e)
            raise

        return self.numberOfTopics,self.Coherence,self.Perplexity

    def assign_Topic_To_Tweet(self,preProcessedData_BOW_df):

        print("Starting: assign_Topic_To_Tweet")
        preProcessedData_BOW_df_withTopics = assign_Topic_To_document(preProcessedData_BOW_df=preProcessedData_BOW_df,
                                                                      winningModel=self.winningLDAModel,
                                                                      corpus=self.corpus,
                                                                      id2word=self.id2word,
                                                                      numberOfTopics=self.numberOfTopics,
                                                                      dataFrameWithText=self.dataFrameWithText,
                                                                      textColumnName=self.textColumnName,
                                                                      printResults=self.printResults
                                                                      )

        print("Finished: assign_Topic_To_Tweet")
        return preProcessedData_BOW_df_withTopics


    def save_model_Pikcle(self):
        pickle.dump(self.winningLDAModel, open(self.path_LDAModel, "wb"))
        pickle.dump(self.corpus, open(self.path_Corpus, "wb"))
        pickle.dump(self.id2word, open(self.path_Id2Words, "wb"))
        pickle.dump(self.mainCorpusFeatures, open(self.path_mainCorpusFeatures, "wb"))
        self.winningLDAModel.save(self.path_LDAModel)
        pd.DataFrame(self.wordsAndTheirTopics).T.to_csv(self.path_WordsToTopicList,index_label="word")
        self.ldaResultOutput_df.to_csv(self.path_LDATrainingOutput,index_label="modelName")

        print('Saved data to %s' % (self.baseSavePath + "LDAModel.pkl"))

    def load_model_Pikcle(self):
        self.winningLDAModel = pickle.load(open(self.path_LDAModel, "rb"))
        self.corpus = pickle.load(open(self.path_Corpus, "rb"))
        self.id2word = pickle.load(open(self.path_Id2Words, "rb"))
        self.mainCorpusFeatures = pickle.load(open(self.path_mainCorpusFeatures, "rb"))
        self.winningLDAModel = gensin_models.LdaModel.load(self.path_LDAModel)
        self.wordsAndTheirTopics = pd.read_csv(filepath_or_buffer=self.path_WordsToTopicList,index_col="word").T.to_dict()
        self.ldaResultOutput_df = pd.read_csv(filepath_or_buffer=self.path_LDATrainingOutput)
        _, self.numberOfTopics,self.Coherence,self.Perplexity = pd.DataFrame(self.ldaResultOutput_df).sort_values(by=['Coherence'], ascending=False).filter(items=['ActualModel', 'TopicNum','Coherence','Perplexity']).head(1).values[0]

    def predict_On_Unseen_Corpus(self, new_dataFrameWithText,randomUserAndTheirTweets_df=None):

        dynamicTopicModel_new,listOfCorpusFeatures = clean_df_text_from_nounsAndAdj(dataFrameWithText=new_dataFrameWithText,
                                                           textColumnName=self.textColumnName,
                                                           applyStemming=self.applyStemming,
                                                           minCountThreshold=0,
                                                           maxCountThreshold=1000)

        # Take BOW, remove redundent features, convert to TDM (Transpose)
        print("Number of feature of new dataset: %s" % (str(dynamicTopicModel_new.shape)))
        new_termDocumentMatric = dynamicTopicModel_new.filter(items=self.mainCorpusFeatures)#.T
        print("Number of feature of new dataset: %s" % (str(new_termDocumentMatric.shape)))

        self.new_dataFrameWithText = new_dataFrameWithText
        self.new_termDocumentMatric = new_termDocumentMatric

        #new_Corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(new_termDocumentMatric))

        new_Corpus, id2word_new, twitsAndTheirUniqueWordList_new = dynamicTopicModel_ToCorpus(new_termDocumentMatric)

        self.new_Corpus = new_Corpus

        randomUserAndTheirTweets_df_withTopics = assign_Topic_To_document(preProcessedData_BOW_df=randomUserAndTheirTweets_df,
                                                    winningModel=self.winningLDAModel,  # Use previsouly trained LDA model
                                                 corpus=new_Corpus,  # Use the new corpus
                                                 id2word=self.id2word,  # Use existing Id2Word dictionary
                                                 dataFrameWithText=new_dataFrameWithText,
                                                 numberOfTopics=self.numberOfTopics,
                                                 textColumnName=self.textColumnName)  # Use new dataFrameWithText

        self.randomUserAndTheirTweets_df_withTopics = randomUserAndTheirTweets_df_withTopics
        self.new_Corpus = new_Corpus

        return randomUserAndTheirTweets_df_withTopics, new_Corpus

    def print_corpus(self, numOfTxts):

        for i, corp in enumerate(self.corpus):
            if i <= numOfTxts:
                #(self.corpus)
                print([(self.id2word[item[0]], item[1]) for item in self.corpus[i]])

    def plot_pyLDAvis(self):

        # Visualize the topics
        pyLDAvis.enable_notebook()
        LDAvis_prepared = pyLDAvis.gensim.prepare(self.winningLDAModel, self.corpus, self.id2word)
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

    peopleAndTheirTweets_df = pd.read_csv(test_processedData,index_col=0)
    updatedStopWords = pickle.load(open(test_updatedStopwords, "rb"))

    listOfRawTweets_df = pd.DataFrame(peopleAndTheirTweets_df['full_tweet'])

    ldaPredictionModel = LDA_Perdictions(baseFolder_models=baseFolder_models_test)

    ldaPredictionModel.main_LDA_function_textToTopics(dataFrameWithText=listOfRawTweets_df, textColumnName='full_tweet', applyStemming=False,
                                                      minCountThreshold=minCountThreshold, maxCountThreshold=maxCountThreshold, useStopWords=True,
                                                      trainNewLDAModelRegardless=shouldTrainNewLDAModelRegardless, trainSeveralLDAModels=False,
                                                      topicList=topicList, printResults=False

                                                      )

    peopleAndTheirTweets_df_withTopics = ldaPredictionModel.assign_Topic_To_Tweet(preProcessedData_BOW_df=peopleAndTheirTweets_df)

    #print(peopleAndTheirTweets_df_withTopics)

    ldaPredictionModel.print_corpus(10)
    #ldaPredictionModel.plot_pyLDAvis()
