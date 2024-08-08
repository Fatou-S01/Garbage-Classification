import base64
from flask import Flask, jsonify
from flask_restful import Api, Resource,request
import pickle
import json
from flask_cors import CORS
import os
import os.path
from os import path
from functions import save_base64_image

  
app = Flask(__name__)
CORS(app)



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/classifyGarbagge", methods=['POST'])
def classifyGarbagge():
     
     data = request.get_json()
     garbaggeImageEncoded= data.get('garbaggeImage')
     garbaggeImageDecoded = base64.b64decode(garbaggeImageEncoded)
     save_base64_image(garbaggeImageDecoded,"garbaggeImage","garbaggePhoto.png")
    # Load modele et classification: UTILISER LA FONCTION PREDICT DE functions.py
    
     return "Image: saved"





if __name__ == "__main__":
    app.run()
