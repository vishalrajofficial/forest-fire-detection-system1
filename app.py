import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict_proba(final_features)

        output = prediction[0][1]

        if output > 0.5:
            return render_template('predict.html', pred=f'You are in DANGER, Probability of fire is {output:.2f}')
        else:
            return render_template('predict.html', pred=f'You are SAFE, Probability of fire is {output:.2f}')
    except Exception as e:
        return render_template('predict.html', pred=f'Error processing input: {e}')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
