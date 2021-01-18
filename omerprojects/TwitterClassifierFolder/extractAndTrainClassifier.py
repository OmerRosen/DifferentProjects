from omerprojects.TwitterClassifierFolder.pythonFiles.DataPreProcessing import basic_preProcessing_BOW_FeatureRemoval
from omerprojects.TwitterClassifierFolder.pythonFiles.Scraper import extractTweetsForListOfUsers
from omerprojects.TwitterClassifierFolder.pythonFiles.NLP_LDA_Class import LDA_Perdictions
from omerprojects.TwitterClassifierFolder.pythonFiles.Classifier_Basic import trainBasicClassifier
import os
import json



def classifyPeopleOfInterest(personsOfInterestList,tweetsPerPerson,baseProjectName='FinalProject',shouldCollectComments=False,
                             extractTwitsRegardlessIfExists = False,
                             minThresholdForFeature=30,maxThresholdForFeature = 150,featureReduction_NumberOfFeaturesToLeave=15000,
                             shouldPerformDataPreProcessingRegardless = False,shouldTrainNewLDAModelRegardless = False,
                             topicList=[3,4,5,7,9],shouldLoadPretrainedStopwordList = True,shouldTrainClassifiersRegardless=True,trainNewModelRegardless=True):


    #Fixed parameters
    shouldCollectVocabularyWord = True
    minimumNumberOfWordPerTweet = 10
    shouldPerformNgram = True


    personsOfInterestDict = {i:name.lower() for i, name in enumerate(personsOfInterestList) }
    personsOfInterestDict_Reversed = {name.lower():i for i, name in enumerate(personsOfInterestList) }

    projectName = '%s_%sClasses_%stwtpp' %(baseProjectName,len(personsOfInterestList),tweetsPerPerson)
    NLPMethodsString = "_"

    if shouldPerformNgram: NLPMethodsString+="nGrams_"

    nameShortcuts= "_".join([person[:2] for person in personsOfInterestList])
    extandedProjectName = "%s_authors_%s"%(projectName,nameShortcuts)

    baseFolder = os.path.dirname(os.path.realpath(__file__))
    print('dir_path')
    print(baseFolder)


    baseFolder_static = os.path.join(baseFolder, 'static')
    baseFolder_models = os.path.join(baseFolder, 'models', extandedProjectName)
    baseFolder_LDA = os.path.join(baseFolder_models, 'LDA')
    baseFolder_Classifier = os.path.join(baseFolder_models, 'Classifier')

    if not os.path.exists(baseFolder_static):
        os.makedirs(baseFolder_static)
    if not os.path.exists(baseFolder_models):
        os.makedirs(baseFolder_models)

    if not os.path.exists(baseFolder):
        os.makedirs(baseFolder)


    completeListOfPeopleAndTheirTweets,personsOfInterest_extandedDetails = extractTweetsForListOfUsers(
                                                                                                personsOfInterestList=personsOfInterestList,
                                                                                                baseFolder_models=baseFolder_models,
                                                                                                baseFolder_static = baseFolder_static,
                                                                                                extractTwitsRegardlessIfExists=extractTwitsRegardlessIfExists,
                                                                                                tweetsPerPerson=tweetsPerPerson,
                                                                                                minimumNumberOfWordPerTweet=minimumNumberOfWordPerTweet,
                                                                                                shouldCollectComments=shouldCollectComments,
                                                                                                shouldSaveAsCSV=True)


    peopleAndTheirTweets_df,stopWordsList = basic_preProcessing_BOW_FeatureRemoval(completeListOfPeopleAndTheirTweets=completeListOfPeopleAndTheirTweets,
                                                                                   baseFolder_static=baseFolder_static,
                                                                                   baseFolder_models=baseFolder_models,
                                                                                   minThresholdForFeature=minThresholdForFeature,
                                                                                   maxThresholdForFeature=maxThresholdForFeature,
                                                                                   featureReduction_NumberOfFeaturesToLeave=featureReduction_NumberOfFeaturesToLeave,
                                                                                   shouldPerformDataPreProcessingRegardless=shouldPerformDataPreProcessingRegardless)


    ldaPredictionModel = LDA_Perdictions(baseFolder_models=baseFolder_LDA)

    lda_numberOfTopics,lda_Coherence,lda_Perplexity = ldaPredictionModel.main_LDA_function_textToTopics(dataFrameWithText=peopleAndTheirTweets_df,
                                                      textColumnName='full_tweet',
                                                      applyStemming=False,
                                                      minCountThreshold=minThresholdForFeature,
                                                      maxCountThreshold=maxThresholdForFeature,
                                                      useStopWords=True,
                                                      trainNewLDAModelRegardless=False,
                                                      trainSeveralLDAModels=False,
                                                      topicList=topicList,
                                                      printResults=False

                                                      )

    peopleAndTheirTweets_df_withTopics = ldaPredictionModel.assign_Topic_To_Tweet(preProcessedData_BOW_df=peopleAndTheirTweets_df)

    print("Starting: trainBasicClassifier")
    classifierModel,headersForTraining,model_score,model_val_score = trainBasicClassifier(dataSet=peopleAndTheirTweets_df_withTopics,
                                           lableColumnName='aa_UserName',
                                           classDict_Reversed=personsOfInterestDict_Reversed,
                                           classifierBasePath=baseFolder_Classifier,
                                           shouldTrainModelRegardless=True)
    print("Starting: trainBasicClassifier")


    numberOfFeatures = len(headersForTraining)

    modelParameterJson = {
        "personsOfInterestList":personsOfInterestList,
        "personsOfInterestDict":personsOfInterestDict,
        "personsOfInterestDict_Reversed":personsOfInterestDict_Reversed,
        "baseProjectName":baseProjectName,
        "tweetsPerPerson":tweetsPerPerson,
        "shouldCollectComments":shouldCollectComments,
        "extractTwitsRegardlessIfExists":extractTwitsRegardlessIfExists,
        "shouldPerformDataPreProcessingRegardless":shouldPerformDataPreProcessingRegardless,
        "shouldTrainNewLDAModelRegardless":shouldTrainNewLDAModelRegardless,
        "shouldTrainClassifiersRegardless":shouldTrainClassifiersRegardless,
        "shouldCollectComments":shouldCollectComments,
        "shouldPerformNgram":shouldPerformNgram,
        "minThresholdForFeature":minThresholdForFeature,
        "maxThresholdForFeature":maxThresholdForFeature,
        "featureReduction_NumberOfFeaturesToLeave":featureReduction_NumberOfFeaturesToLeave,
        "minimumNumberOfWordPerTweet":minimumNumberOfWordPerTweet,
        "shouldLoadPretrainedStopwordList":shouldLoadPretrainedStopwordList,
        "topicList":topicList,
        "trainNewModelRegardless":trainNewModelRegardless,
        'outputs':{
            "extandedProjectName":extandedProjectName,
            "personsOfInterest_extandedDetails":personsOfInterest_extandedDetails.T.to_dict(),
            "lda_numberOfTopics":lda_numberOfTopics,
            "lda_Coherence":lda_Coherence,
            "lda_Perplexity":lda_Perplexity,
            "numberOfFeatures":numberOfFeatures,
            "model_score":model_score,
            "model_val_score":model_val_score,

        }

    }

    parameterJson_path = os.path.join(baseFolder_models,'Model_HyperParameters.txt')
    with open(parameterJson_path, 'w') as outfile:
        json.dump(modelParameterJson, outfile)

    return modelParameterJson


if __name__=="__main__":

    # Correct amount of classes for this project (In the future might attempt more)
    personsOfInterestList = ['JoeBiden', 'ladygaga', 'IAM_Shakespeare', 'iamcardib', 'ElonMusk', 'StephenKing','karpathy']
    personsOfInterestList = ['amandapalmer','taylorswift13','jtimberlake','andersoncooper','maddow']

    # Parameters:

    baseProjectName = 'FinalProject'
    tweetsPerPerson = 1000
    shouldCollectComments = True # Collect user's comments in addition to tweets

    # Parameter regarding skipping steps in processing:
    extractTwitsRegardlessIfExists = False #If false, will check if tweets for these users were previouly save, and load them instead of scraping
    shouldProcessRawData = True #If false, will check if raw BOW dataset was previouly save, and load it
    shouldPerformDataPreProcessingRegardless = False
    shouldTrainNewLDAModelRegardless = False #If false, will check if LDA model was previously saveed, and load it
    shouldTrainClassifiersRegardless = True  #If false, will check if classifier model was previously saveed, and load it

    # Hyper Parameters:
    shouldPerformNgram = True #If false, will skip nGram process
    minThresholdForFeature = 30 # int(tweetsPerPerson*0.04)   # Remove features that appear in less than N % of the corpus
    maxThresholdForFeature = 150  # Remove features that appear in more than N % of the corpus
    featureReduction_NumberOfFeaturesToLeave = 20000 # After sorting feature importance, how many top features to leave (None = leave all)
    minimumNumberOfWordPerTweet = 10    #Twits with less than N words are not worth of analysing
    shouldLoadPretrainedStopwordList = True #In case list of stopwords was already collected form previous run - Use it
    topicList=[3,4,5,7,9]
    trainNewModelRegardless = False



    modelParameterJson = classifyPeopleOfInterest(personsOfInterestList=personsOfInterestList, baseProjectName=baseProjectName, tweetsPerPerson=tweetsPerPerson, shouldCollectComments=shouldCollectComments,
                             extractTwitsRegardlessIfExists=extractTwitsRegardlessIfExists,
                             minThresholdForFeature=minThresholdForFeature, maxThresholdForFeature=maxThresholdForFeature,featureReduction_NumberOfFeaturesToLeave=featureReduction_NumberOfFeaturesToLeave,
                             shouldPerformDataPreProcessingRegardless=shouldPerformDataPreProcessingRegardless, shouldTrainNewLDAModelRegardless=shouldTrainNewLDAModelRegardless,
                             topicList=topicList, shouldLoadPretrainedStopwordList=True,shouldTrainClassifiersRegardless = shouldTrainClassifiersRegardless,trainNewModelRegardless=trainNewModelRegardless)