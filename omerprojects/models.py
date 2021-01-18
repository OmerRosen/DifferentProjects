from datetime import  datetime
from omerprojects import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userName = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), unique=True, nullable=False, default='default.jpg')
    password = db.Column(db.String(60), unique=True, nullable=False)

    tweets = db.relationship('Tweet',backref='author', lazy=True)

    reg_date = db.Column(db.DateTime(60), nullable=False, default=datetime.utcnow )

    def __repr__(self):
        return "User( %s, %s, %s) %({self.username},{self.email},{self.image_file})"

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tweetId = db.Column(db.Integer)
    full_text = db.Column(db.String())
    tweetDate = db.Column(db.DateTime)
    wordCount = (db.Integer)
