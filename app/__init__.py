from flask import Flask
from flask_restful import Resource, Api
from flask_bootstrap import Bootstrap

app = Flask(__name__)
api = Api(app)

app.config['SECRET_KEY'] = "YOUR SECRET KEY"
bootstrap = Bootstrap(app)

from app import routes

