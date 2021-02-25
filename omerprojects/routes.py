from flask import render_template, url_for, flash, redirect,request
from omerprojects.forms import RegistrationForm, LoginForm, CollectTwitterUsers_ForTraining,InputYourTwitterIdForClassification,NudityDetectorForm
from omerprojects import app
from omerprojects.TwitterClassifierFolder.pythonFiles.CollectAllAvailableModels import gatherAllAvailableModels
from omerprojects.TwitterClassifierFolder.extractAndTrainClassifier import classifyPeopleOfInterest
from omerprojects.TwitterClassifierFolder.whichTwitterPricessAreYou import analyseTweetsForRandomUser
from omerprojects.NudityDetector.MakePredictionUsingWinningModel import takeImagePath_ReturnPredictions
from werkzeug.utils import secure_filename

import os
import pandas as pd

import threading

@app.route("/")
@app.route("/home")
def home():
    #return "<h1>Hello world!</h1>"
    return render_template('home.html')

@app.route("/about")
def about():
    #return "<h1>Hello world!</h1>"
    return render_template('about.html')

@app.route("/register", methods=['GET','POST'])
def register():
    form = RegistrationForm()
    isValidate = form.validate_on_submit()
    if isValidate:
        flash("Account Created For %s!"%({form.username.data}), 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET','POST'])
def login():
    form = LoginForm()
    isValidate = form.validate_on_submit()
    if isValidate:
        if form.email.data == 'rosen.omer@gmail.com' and form.password.data=='1234qwer!':
            flash('You have been logged in!', category='success')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful!', category='danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/twitterClassifier_Train", methods=['GET','POST'])
def twitterClassifier_Train():
    form = CollectTwitterUsers_ForTraining()
    isValidate = form.validate_on_submit()
    personsOfInterest_html = None
    if isValidate:
        personsOfInterestList = form.personsOfInterestList.data.split(',')
        tweetsPerPerson = form.tweetsPerPerson.data
        shouldCollectComments = form.shouldCollectComments.data
        shouldTrainFromStart = form.shouldTrainFromStart.data

        modelParameterJson = classifyPeopleOfInterest(personsOfInterestList=personsOfInterestList, baseProjectName='FinalProject',
                                 tweetsPerPerson=tweetsPerPerson, shouldCollectComments=shouldCollectComments,
                                 extractTwitsRegardlessIfExists=shouldTrainFromStart,
                                 minThresholdForFeature=15,
                                 maxThresholdForFeature=150,
                                 featureReduction_NumberOfFeaturesToLeave=15000,
                                 shouldPerformDataPreProcessingRegardless=shouldTrainFromStart,
                                 shouldTrainNewLDAModelRegardless=shouldTrainFromStart,
                                 topicList=[3,4,6,7], shouldLoadPretrainedStopwordList=True)

        personsOfInterest_dict = modelParameterJson['outputs']['personsOfInterest_extandedDetails']
        lda_numberOfTopics = modelParameterJson['outputs']['lda_numberOfTopics']
        lda_Coherence = modelParameterJson['outputs']['lda_Coherence']
        numberOfFeatures = modelParameterJson['outputs']['numberOfFeatures']
        model_val_score = modelParameterJson['outputs']['model_val_score']
        flash(
            'Process Completed:\n Validation Score: %s, Feature Count: %s, LDA Topic Num: %s, LDA Coherence: %s'%(round(model_val_score,2),numberOfFeatures,lda_numberOfTopics,round(lda_Coherence,2))
            , category='success')

        return render_template('twitterClassifier_ForTraining.html', title='Twitter Classifier Training', form=form, personsOfInterest_dict=personsOfInterest_dict)

    return render_template('twitterClassifier_ForTraining.html', title='Twitter Classifier Training', form=form)


@app.route("/InputYourTwitterIdForClassification", methods=['GET','POST'])
def inputYourTwitterIdForClassification():

    basefolder = os.path.join(os.getcwd(),"omerprojects/TwitterClassifierFolder")
    #listOfModels = gatherAllAvailableModels(basefolder=basefolder)
    listOfModels = gatherAllAvailableModels(os.path.join(basefolder,'Models'))

    ddlOptions = []

    for model in listOfModels:
        try:
            userNamesAndNameList = model['outputs']['personsOfInterest_extandedDetails']['name']
        except Exception as e:

            userNamesAndNameList = pd.DataFrame(model['outputs']['personsOfInterest_extandedDetails']).T
            userNamesAndNameList = userNamesAndNameList.to_dict()['name']

        try:
            userNamesAndNameList_str = " - ".join([name for userName, name in userNamesAndNameList.items()])
            tweetsPerPerson = model['tweetsPerPerson']

            extandedProjectName = model['outputs']['extandedProjectName']
            model_score = model['outputs']['model_score']
            model_val_score = model['outputs']['model_val_score']

            modelStringDesc = "%s accuracy, %s authors, %s tweets each. Users: %s"%(round(model_val_score,2),len(userNamesAndNameList),tweetsPerPerson,userNamesAndNameList_str)

            ddlOptions.append((extandedProjectName,modelStringDesc))
        except Exception as e:
            print(model)
            print(e)

    form = InputYourTwitterIdForClassification()

    form.chooseYourModel.choices=ddlOptions

    isValidate = form.validate_on_submit()

    if isValidate:
        randomUser_extandedDetails = form.username.data.split(',')
        selectedModel = form.chooseYourModel.data
        shouldCollectComments = form.shouldCollectComments.data
        try:
            twitsAndTheirSimilarity_summary, randomUser_extandedDetails, personsOfInterest_extandedDetails = analyseTweetsForRandomUser(
                randomUsername=randomUser_extandedDetails, extandedProjectName=selectedModel,
                shouldCollectComments=shouldCollectComments, fontSize_px=22,
                nTweetsToCompareFromAuthor=300,baseFolder=basefolder)

            flash(
                'Process Completed'
                , category='success')

            return render_template('InputYourTwitterIdForClassification.html', title='Twitter Classifier Training',
                                   form=form,
                                   twitsAndTheirSimilarity_summary=twitsAndTheirSimilarity_summary.head(3),
                                   randomUser_extandedDetails=randomUser_extandedDetails,
                                   personsOfInterest_extandedDetails=personsOfInterest_extandedDetails,
                                   testParam=True
                                   )
        except Exception as e:
            flash(
                'Process Failed: %s'%(e)
                , category='warning')



    return render_template('InputYourTwitterIdForClassification.html', title='Twitter Classifier', form=form)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/nudityDetector", methods=['GET','POST'])
def nudityDetector():
    form = NudityDetectorForm()
    try:
        if request.method == 'POST':

            listOfImagePaths = []
            files = request.files.getlist("file")
            limitCount = 10
            for file in files:
                if limitCount > 0:
                    if allowed_file(file.filename):
                        originalFileNAme = file.filename
                        safeFileName = secure_filename(file.filename)
                        absoluteSavePath = os.path.join(app.config['UPLOAD_FOLDER'],safeFileName)
                        relativeSavePath = os.path.join('static','uploads',safeFileName)
                        file.save(absoluteSavePath)
                        distList = {"originalFileNAme":originalFileNAme,
                                    "safeFileName":safeFileName,
                                    "absoluteSavePath":absoluteSavePath,
                                    "relativeSavePath":relativeSavePath,
                                    }
                        listOfImagePaths.append(absoluteSavePath)
                        limitCount -= 1
                else:
                    break

            winningModelPath = r"C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\NudityDetector\Models\ModelOutput - NudiyDetector_Draft2\NudiyDetector_Draft2 - Accuracy 0.6.hdf5"
            imageListClassification = takeImagePath_ReturnPredictions(imagesPathList=listOfImagePaths,
                                                                      requestedModelAbsPath=winningModelPath)




            flash(
                'Process Completed. %s files were retrieved'%(len(files))
                , category='success')

            return render_template('NudityDetector.html', title='Nudity detector',
                                   form=form, imageListClassification=imageListClassification, showFiles=True
                                   )
    except Exception as e:
        flash(
            'Process Failed. Error: %s ' % (e)
            , category='warning')


    return render_template('NudityDetector.html', title='Nudity Detector', form=form, showFiles=False)





