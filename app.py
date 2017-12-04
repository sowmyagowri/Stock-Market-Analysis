from flask import Flask, render_template, request, make_response, url_for, flash, redirect, session, abort, jsonify,g
import json
from jsonschema import validate, ValidationError
# from flask_httpauth import HTTPBasicAuth
from itsdangerous import (TimedJSONWebSignatureSerializer as Serializer, BadSignature, SignatureExpired)
from functions import *
import  myexception

app = Flask(__name__)

# auth = HTTPBasicAuth()

@app.errorhandler(myexception.MyExceptions)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

#home route
@app.route('/')
def index():
    return render_template('index.html')

#stock prices trend api
@app.route('/prediction', methods=['GET', 'POST'])
def get_prediction():

     print (request.json)
     company = request.json
     print (company)
     result = predict_prices(company)
     print(result)
     return json.dumps(result)

if __name__ == '__main__':
    app.run(debug = True, host='localhost', threaded=False, port=8080)
