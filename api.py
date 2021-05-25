"""
Gintautas Plonis 1812957
Electra | SQuAD 2.0
(Optional) REST API
"""
import json

import flask as flask
from flask import request

from predict_single import predict

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def index():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/predict', methods=['POST'])
def post():
    content = request.get_json(silent=True)
    if content is None or 'question' not in content or 'context' not in content or 'id' not in content:
        return json.dumps({'message': 'Payload unsupported.'}), 400, {'ContentType': 'application/json'}

    answer = predict(question=content['question'], context=content['context'], id=content['id'], model='model_8')

    return json.dumps({'answer': answer}), 200, {'ContentType': 'application/json'}


app.run()
