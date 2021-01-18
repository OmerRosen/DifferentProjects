from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import  datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'uCzzBfb4qUvIkYRehbJn7KbZZTED2FeG'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


from omerprojects import routes