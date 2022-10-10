import pickle
from flask import Flask
from flask import jsonify
from flask import request

# load model
model_file_path = 'model2.bin'
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

# load DictVectorizer
dv_file_path = 'dv.bin'
with open(dv_file_path, 'rb') as dv_file:
    dv = pickle.load(dv_file)

# create Flask application instance:
app = Flask('card')

@app.route('/predict', methods=['POST'])


def predict():
    client = request.get_json()
    x_test = dv.transform(client)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    result = {
        'card_approved_probability': float(y_pred_proba)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)