import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random as rand
import numpy as np
import string
import keras
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


intents = ["greeting", "bye", "view_entry", "add_entry"]

def responses_list():
    data1 = json.loads(open('responses.json').read())
    responses_list = {i: [] for i in intents}


    for inputs in data1:
        responses_list[inputs['intent'].lower()].append(inputs["response"])
    return responses_list

def load_data():
    data1 = json.loads(open('intents.json').read())
    tags = []
    inputs_list = []
    for inputs in data1:
        inputs_list.append(inputs["Inputs"])
        tags.append(inputs["Intents"])
    data = pd.DataFrame({'inputs': inputs_list, 'tags': tags})
    data['inputs'] = data['inputs'].apply(lambda w: [l.lower() for l in w if l not in string.punctuation])
    data['inputs'] = data['inputs'].apply(lambda w: ''.join(w))
    tokenwords = Tokenizer(num_words=2000)
    tokenwords.fit_on_texts(data['inputs'])
    labelencoding = LabelEncoder()
    xy = labelencoding.fit_transform(data['tags'])
    return tokenwords, labelencoding

def get_intent(user_input):
    tokenwords, labelencoding = load_data()
    model = keras.models.load_model('intent_classifier.h5')
    list_text = []
    make_prediction = [letters.lower() for letters in user_input if letters not in string.punctuation]
    make_prediction = ''.join(make_prediction)
    list_text.append(make_prediction)
    make_prediction = tokenwords.texts_to_sequences(list_text)
    make_prediction = np.array(make_prediction).reshape(-1)
    make_prediction = pad_sequences([make_prediction], 16)
    lstm_p = model.predict(make_prediction)
    lstm_p = lstm_p.argmax()
    preidicted_response_tag = labelencoding.inverse_transform([lstm_p])[0]

    return preidicted_response_tag

def get_response_by_intent(intent):
    responses = responses_list()
    return rand.choice(responses[intent])



