from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import  datetime

UPLOAD_FOLDER = r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'uCzzBfb4qUvIkYRehbJn7KbZZTED2FeG'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)


from omerprojects import routes