from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, SelectField
from wtforms.validators import DataRequired,Length, Email, EqualTo
import email_validator
from flask import Flask, flash, request, redirect, url_for


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(),Length(min=4,max=50)]
                           )

    email = StringField('Email', validators=[DataRequired(), Email() ])

    password = PasswordField('Password',
                             validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(),EqualTo('password')])


    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):

    email = StringField('Email', validators=[DataRequired(), Email() ])

    password = PasswordField('Password',
                             validators=[DataRequired()])

    remember = BooleanField('Remember me')

    submit = SubmitField('Login')


class CollectTwitterUsers_ForTraining(FlaskForm):
    baseProjectName = StringField('BaseProjectName',
                           validators=[DataRequired(),Length(min=4,max=50)],
                           default='DefaultProject'
                           )

    tweetsPerPerson = IntegerField('TweetsPerPerson', validators=[DataRequired()], default=20)

    shouldCollectComments = BooleanField('Collect user comments as well')

    shouldTrainFromStart = BooleanField('Should Train model regardless is exists already?')


    personsOfInterestList = StringField('PeopleOfInterest',
                           validators=[DataRequired()],
                           default='JoeBiden,ladygaga,iamcardib,ElonMusk,StephenKing'
                           )


    submit = SubmitField('Extract Tweets')

class InputYourTwitterIdForClassification(FlaskForm):

    username = StringField('Your Twitter Username',
                           validators=[DataRequired(),Length(min=4,max=50)]
                           )

    chooseYourModel = SelectField(label='Choose your model', choices=[])

    shouldCollectComments = BooleanField('Collect user comments as well')

    submit = SubmitField('Which Twitter princess are you?')





class NudityDetectorForm(FlaskForm):

    submit = SubmitField('Search for nudes')
