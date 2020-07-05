"""
Author: Ahad Suleymanli

"""

from flask import Flask, json
from flask_cors import CORS, cross_origin
from flask import request
from PIL import Image
import base64
import io
import numpy as np
import matplotlib.pyplot as plt

import inference_server as _inference_server

inference_server = _inference_server.InferenceServer()

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'

@api.route('/', methods=['POST'])
@cross_origin()
def post_image():
    raw_string = request.data
    x = raw_string[raw_string.find(b'/9'):]
    try:
        base64_decoded = base64.b64decode(x)
        image = Image.open(io.BytesIO(base64_decoded))
    except:
        return json.dumps({"result": "incorrect image format"}), 201
        
    image_np = np.array(image)
    result = inference_server.classify(image_np)
    return json.dumps({"result": result}), 201

@api.route('/changemodel', methods=['POST'])
@cross_origin()
def change_model():
    model_name = request.data.decode("utf-8")
    inference_server.request_model_change(model_name)
    return json.dumps({}), 201

if __name__ == '__main__':
    api.run(port=3001)