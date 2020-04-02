from flask import Flask,render_template
from flask_restful import Api, Resource, reqparse
import pickle

app = Flask(__name__)
api = Api(app)

class Model(Resource):
    def get(self, features):
        model = pickle.load(open("model.pickle","rb"))   
        pred_flower = model.predict([features])
        return pred_flower[0], 200

api.add_resource(Model, "/<string:features>")
app.run()