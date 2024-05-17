from flask import Flask, request # type: ignore
import joblib
import pandas as pd # type: ignore

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return {'prediction': prediction[0]}

if __name__ == '__main__':
    app.run(port=5000)
