import joblib
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
import joblib
import re
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')


# Loading tokenizer and lgbm model
VECTORIZER = joblib.load('tfidf_vectorizer.sav')
MODEL = joblib.load('model.pkl')

# Sample Input

# positive review
# sample = "Fantastic and clean version of melatonin - fast acting and doesn’t give me nightmares like other brands have - been using for two years - great value here"

# negative review
sample = "Took these for the first time last night, up for 12 hours with horrible stomach pains. I wouldn’t recommend these to my worst enemies"


# Function to preprocess text
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

# Predict Function
def predict(text=''):
    # preprocess text
    vector = preprocess_text(text)
    # make prediction
    prediction = MODEL.predict(vector)[0]
    if prediction == 0:
        print(f'Review: {text}\nSentiment: Bad Review!')
    else:
        print(f'Review: {text}\nSentiment: Good Review!')
    return prediction

### INSERT YOUR CUSTOM REVIEW HERE ###
# sample = "insert here and remove the hashtag"

predict(sample)