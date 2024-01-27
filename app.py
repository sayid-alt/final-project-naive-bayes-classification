from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

gnb_model = joblib.load(open('gnb_model.pkl', 'rb'))
sc = joblib.load(open('robust_scaler.pkl', 'rb'))


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    age = int(request.form.get('age'))
    salary_idn = int(request.form.get('salary'))
    salary_usd = int(salary_idn / 15776)

    # 2 dimensional array of features
    features_2d = np.array([[age, salary_usd]])

    # features scaled
    final = sc.transform(features_2d)

    # predict
    prediction = gnb_model.predict(final)

    if prediction == 1:
        return render_template('predict.html', pred='The guy is going to purchase the insurrance', status='Positive')
    else:
        return render_template('predict.html', pred='The guy ain\'t going to purchase the insurrance', status='Negative')


if __name__ == '__main__':
    app.run(debug=True)
