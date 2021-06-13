from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask_cors import CORS
import numpy as np
#import pandas as pd
import tensorflow as tf
import re, os, gc
import pickle

app = Flask(__name__)
CORS(app)

def set_app():
    file = open('assets/idx_to_word.txt', 'rb')
    idx_to_word = pickle.load(file)
    #print('idx_to_word loaded...')
    file.close()
    file = open('assets/word_to_idx.txt','rb')
    word_to_idx = pickle.load(file)
    #print('word_to_idx loaded...')
    file.close()
    json_file = open('assets/best_model_base.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("assets/best_model_base.hdf5")
    #print("Loaded model from disk")
    return (idx_to_word, word_to_idx, loaded_model)

idx_to_word, word_to_idx, loaded_model = set_app()

def postprocessing(review):
    review = review.lower()
    review_arr = re.split('[^A-Za-z]+', review)
    review_final = []
    for i in review_arr:
        if len(i) > 1:
            review_final.append(i)
    return np.array(review_final)

def sentences_to_indices(word_to_idx, X, maxlen = 150):
    m = X.shape[0]
    indices = np.zeros((1, maxlen))
    j = 0
    for word in X:
        if word_to_idx.get(word) is not None:
            indices[0, j] = word_to_idx[word]
        j += 1
    return indices


def predict(loaded_model, review_indices):
    return np.argmax(loaded_model.predict(review_indices), axis = 1)


@app.route('/', methods = ['POST', 'GET'])
def home():
    return render_template('index.html')
@app.route('/test', methods = ['POST', 'GET'])
def test():
    return render_template('Test_api.html')

@app.route('/predict', methods = ['POST', 'GET'])
def pipeline_predict(word_to_idx = word_to_idx, loaded_model = loaded_model):
    review = request.form['nm']
    review_proc = postprocessing(review)
    review_indices = sentences_to_indices(word_to_idx, review_proc)
    res = predict(loaded_model, review_indices)
    return "<h1>" + str(int(res)) + "</h1>"

@app.route('/api-predict', methods = ['POST', 'GET'])
def pipeline_predict_and_reply(word_to_idx = word_to_idx, loaded_model = loaded_model):
    review_json = request.json
    review = review_json['review']
    #print(review)
    review_proc = postprocessing(review)
    review_indices = sentences_to_indices(word_to_idx, review_proc)
    res = predict(loaded_model, review_indices)
    response_data = {'score': str(int(res)),'message': 'Created', 'code': 'SUCCESS'}
    return make_response(jsonify(response_data))


if __name__ == "__main__":
    app.run()


