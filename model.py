import os
import joblib
import re
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Defining global variables
MODEL_PATHS = os.listdir('models/')[:-2]
MODEL_PATHS = ['models/'+ path for path in MODEL_PATHS]
MODEL_NAMES = ['bert', 'lgbm', 'lr', 'nb', 'rf']
MODELS_DICT_PATH = dict(zip(MODEL_NAMES, MODEL_PATHS))
VECTORIZER_PATH = r"models/tfidf_vectorizer.sav"
TOKENIZER_PATH = r"models/tokenizer.sav"
VECTORIZER = joblib.load(VECTORIZER_PATH)
TOKENIZER = joblib.load(TOKENIZER_PATH)

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Custom Function for BERT prediction
def predict_bert(model, text):
    input = TOKENIZER(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**input).logits

    predicted_class_id = logits.argmax().item()
    prediction = model.config.id2label[predicted_class_id]
    pred = 0
    if prediction == 'LABEL_0':
        pred = 0
    else:
        pred = 1
    return pred


# Function to preprocess data
def preprocess_text(text):
    
    # converting the text to lower case
    text = text.lower()

    # replacing anything which is a digit and not a word or whitespace with " "
    text = re.sub("[^\w\s]|[\d]", " ", text)

    # stripping excess spaces in text
    text = text.strip()

    # tokenizing
    tokens = word_tokenize(text)

    # stopwords
    stopwords_data = list(set(stopwords.words("english")))
    cleaned_text = []
    for word in tokens:
        if word not in stopwords_data:
            cleaned_text.append(word)

    # joining all the words back
    cleaned_text = " ".join([text for text in cleaned_text])

    # vectorizing the text
    vector = VECTORIZER.transform([cleaned_text])
    vector = vector.toarray()

    return vector

# Sentiment Predict function based on model
def predict(text='', model=''):
    if model == 'bert':
        bert_model = joblib.load(MODELS_DICT_PATH[model])
        prediction = predict_bert(bert_model, text)
    else:
        vector = preprocess_text(text)
        pretrained_model = joblib.load(MODELS_DICT_PATH[model])
        prediction = pretrained_model.predict(vector)[0]
    return prediction
