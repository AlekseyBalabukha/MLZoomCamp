import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])

def predict():
    # get body of request as python dictionary:
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    # our response will also be json:
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    } # dictionary

    return jsonify(result) # turn into json format

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)