#!/usr/bin/env python

from flask import Flask, json, request
from manual import predict

app = Flask(__name__)

@app.route('/')
def index():
    return 'Simple REST service.'

@app.route('/qa', methods=['POST'])
def qa_service():
    if request.headers['Content-Type'] == 'application/json':
        result = predict(request.json)
        return json.dumps(result)
    else:
        return 'Bad request: invalid json'

if __name__ == '__main__':
    app.run(debug=True)

