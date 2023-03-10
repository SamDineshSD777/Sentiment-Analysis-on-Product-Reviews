from flask import Flask, render_template, request
from model import predict
app = Flask(__name__, template_folder='templates')

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':
        model_name = request.form['model']
        text = request.form['text']
        result = predict(text, model_name)
        if int(result) == 1:
            prediction = 'Positive Review!'
        else:
            prediction = 'Negative Review!'
    return render_template("prediction.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)